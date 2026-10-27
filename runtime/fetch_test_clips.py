"""Stream N clips plus transcripts out of one remote OpenSLR 53 zip (HTTP range, no 900 MB download).

Zip f is never used for training, so it is the held-out check:

    python runtime/fetch_test_clips.py --out testclips --n 100 --zip f
"""

import argparse
import csv
import io
import random
import ssl
import urllib.request
import zipfile
from pathlib import Path

try:
    import certifi  # python.org builds ship without a CA bundle
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = None
BASE = "https://www.openslr.org/resources/53/"


def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), context=CTX)


class RemoteFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        self.size = int(get(url, {"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = get(self.url, {"Range": f"bytes={self.pos}-{end}"}).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--zip", default="f")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
text = {r[0]: r[2] for r in csv.reader(io.TextIOWrapper(get(BASE + "utt_spk_text.tsv"), "utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
z = zipfile.ZipFile(io.BufferedReader(RemoteFile(f"{BASE}asr_bengali_{args.zip}.zip"), 1 << 20))
names = sorted(x for x in z.namelist() if x.endswith(".flac"))
with open(args.out / "refs.tsv", "w", encoding="utf-8") as f:
    for name in random.Random(0).sample(names, args.n):
        uid = Path(name).stem
        (args.out / f"{uid}.flac").write_bytes(z.read(name))
        f.write(f"{uid}\t{text[uid]}\n")
        f.flush()  # a killed run keeps the rows it finished
print(f"{len(names)} clips in zip {args.zip}; wrote {args.n} to {args.out}")

"""CER, WER and speed on a folder made by fetch_test_clips.py.

    python runtime/evaluate.py --model Rslt --clips testclips
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import torch
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.bangla_tokenizer import BanglaSyllableTokenizer  # noqa: E402
from model.student import load_student, mel_frames  # noqa: E402


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


assert edit_distance("kitten", "sitting") == 3 and edit_distance("ab".split(), []) == 1

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--clips", type=Path, required=True)
parser.add_argument("--weights", default="student.pt")
parser.add_argument("--show", type=int, default=5)
args = parser.parse_args()
torch.set_num_threads(4)
feat = WhisperFeatureExtractor.from_pretrained(args.model)
tok = BanglaSyllableTokenizer.load(args.model / "bn.syllables.json")
model = load_student(args.model / args.weights)
ce = cn = we = wn = n = 0
audio_s = compute_s = 0.0
for n, line in enumerate((args.clips / "refs.tsv").read_text(encoding="utf-8").splitlines(), 1):
    uid, ref = line.split("\t")
    wav, _ = librosa.load(args.clips / f"{uid}.flac", sr=16000)
    audio_s += len(wav) / 16000
    t = time.perf_counter()
    mel = feat(wav, sampling_rate=16000, return_tensors="pt").input_features[..., : mel_frames(len(wav))]
    hyp = tok.decode(model.generate(mel, tok.bos_id(), tok.eos_id())[0].tolist())
    compute_s += time.perf_counter() - t
    ce += edit_distance(ref.replace(" ", ""), hyp.replace(" ", ""))
    cn += len(ref.replace(" ", ""))
    we += edit_distance(ref.split(), hyp.split())
    wn += len(ref.split())
    if n <= args.show:
        print(f"REF {ref}\nHYP {hyp}\n")
print(f"{args.weights}: {n} clips  CER {ce / cn:.9f}  WER {we / wn:.9f}  "
      f"{audio_s:.0f}s audio in {compute_s:.1f}s  RTF {compute_s / audio_s:.3f}  {compute_s / n * 1000:.0f} ms/clip")

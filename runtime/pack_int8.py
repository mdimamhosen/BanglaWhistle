"""Pack student.pt into student.int8.pt, the ~17 MB file.

    python runtime/pack_int8.py --model /path/to/whistle-bn
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.student import pack_int8  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, help="Output file (defaults to model/student.int8.pt)")
args = parser.parse_args()
out = args.output or args.model / "student.int8.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(pack_int8(torch.load(args.model / "student.pt", map_location="cpu")), out)
print(out, f"{out.stat().st_size / 1e6:.2f} MB")

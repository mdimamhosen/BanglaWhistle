"""Transcribe with the BanglaWhistle rslt3 model.

Train first: kaggle/train.ipynb
The released weights are in checkpoints/rslt3:

    python3 runtime/transcribe.py --model checkpoints/rslt3 --audio clip.wav

Prefers student.int8.pt when present, otherwise uses student.pt; caps input at 30 seconds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.bangla_tokenizer import BanglaSyllableTokenizer  # noqa: E402
from model.student import load_student, mel_frames  # noqa: E402


def load_tokenizer(model_dir: Path):
    syllable = model_dir / "bn.syllables.json"
    if syllable.exists():
        return BanglaSyllableTokenizer.load(syllable)
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(model_dir / "bn.model"))


def transcribe(audio_path: Path, model_dir: Path, weights: str | None = None) -> str:
    feat = WhisperFeatureExtractor.from_pretrained(model_dir)
    sp = load_tokenizer(model_dir)
    weights_path = model_dir / weights if weights else next(
        (model_dir / name for name in ("student.int8.pt", "student.pt") if (model_dir / name).is_file()),
        None,
    )
    if weights_path is None:
        raise FileNotFoundError(f"No student.int8.pt or student.pt found in {model_dir}")
    model = load_student(weights_path)
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_path), "-t", "30", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if decoded.returncode or not decoded.stdout:
        raise RuntimeError("FFmpeg could not decode this audio file")
    speech = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    mel = feat(speech, sampling_rate=16000, return_tensors="pt").input_features
    mel = mel[..., : mel_frames(len(speech))]  # trained on real frames, not 30 s of padding
    ids = model.generate(mel, bos_id=sp.bos_id(), eos_id=sp.eos_id(), beam=5)[0].tolist()
    pieces = [i for i in ids if i not in (sp.bos_id(), sp.eos_id(), sp.pad_id())]
    return unicodedata.normalize("NFC", sp.decode(pieces))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--weights", help="Weight filename (auto-prefers student.int8.pt)")
    args = parser.parse_args()
    print(transcribe(args.audio, args.model, args.weights))


if __name__ == "__main__":
    main()

"""Bangla orthographic-syllable tokenizer.

Pieces are written syllables, not SentencePiece fragments. A hasant keeps the
following consonant in the same piece, and vowel signs stay on that piece.
Ids 0-3 are pad, bos, eos, unk. The embedding table is 4096, so the vocab stops there.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

HASANT = "\u09cd"
ZW = frozenset("\u200c\u200d")
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def syllables(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    buf = ""
    join = False
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
            if out and out[-1] != " ":
                out.append(" ")
            join = False
            continue
        if ord(ch) < 128 and (ch.isalnum() or ch in {"'", "-"}):
            if buf and buf[-1].isascii() and (buf[-1].isalnum() or buf[-1] in {"'", "-"}):
                buf += ch
            else:
                if buf:
                    out.append(buf)
                buf = ch
            join = False
            continue
        cat = unicodedata.category(ch)
        if ch == HASANT:
            buf += ch
            join = True
            continue
        if cat.startswith("M") or ch in ZW:
            buf += ch
            continue
        if join:
            buf += ch
            join = False
            continue
        if buf:
            out.append(buf)
        buf = ch
        join = False
    if buf:
        out.append(buf)
    return out


class BanglaSyllableTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = pieces
        self.to_id = {p: i for i, p in enumerate(pieces)}

    @classmethod
    def train(cls, sentences, vocab_size: int = 4096) -> "BanglaSyllableTokenizer":
        counts: dict[str, int] = {}
        for sentence in sentences:
            for piece in syllables(sentence):
                counts[piece] = counts.get(piece, 0) + 1
        ranked = sorted(counts, key=lambda p: (-counts[p], p))
        keep = ranked[: max(0, vocab_size - 4)]
        return cls(["<pad>", "<s>", "</s>", "<unk>", *keep])

    def extend(self, sentences, vocab_size: int = 4096) -> "BanglaSyllableTokenizer":
        """Add new pieces after the current list. Existing ids stay put, so a checkpoint still matches."""
        counts: dict[str, int] = {}
        for sentence in sentences:
            for piece in syllables(sentence):
                if piece not in self.to_id:
                    counts[piece] = counts.get(piece, 0) + 1
        ranked = sorted(counts, key=lambda piece: (-counts[piece], piece))
        room = max(0, vocab_size - len(self.pieces))
        return type(self)([*self.pieces, *ranked[:room]])

    def encode(self, text: str, out_type: type = int) -> list[int]:
        return [self.to_id.get(p, UNK) for p in syllables(text)]

    def decode(self, ids) -> str:
        parts = []
        for i in ids:
            if i in (PAD, BOS, EOS) or i >= len(self.pieces):
                continue
            parts.append(self.pieces[i])
        return "".join(parts)

    def vocab_size(self) -> int:
        return len(self.pieces)

    def pad_id(self) -> int:
        return PAD

    def bos_id(self) -> int:
        return BOS

    def eos_id(self) -> int:
        return EOS

    def unk_id(self) -> int:
        return UNK

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.pieces, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BanglaSyllableTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

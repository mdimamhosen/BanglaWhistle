"""Bangla speech model sized for a 16.9 MB int8 file.

Weights start random. Quality comes from Bangla transcripts plus the encoder
of a frozen larger Whisper, which is not part of the saved file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FILE_BUDGET_BYTES = int(16.9 * 1_000_000)
D_MODEL = 256
FFN = 1536
HEADS = 4
ENC_LAYERS = 6
DEC_LAYERS = 6
VOCAB = 4096
MAX_TEXT = 256
MEL_BINS = 80


def mel_frames(n_samples: int) -> int:
    """Real mel length of a 16 kHz clip, even so conv2 (stride 2) lines up with pad[:, ::2]."""
    n = min(3000, n_samples // 160 + 1)
    return n + n % 2


def file_bytes(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class StudentASR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = D_MODEL
        self.conv1 = nn.Conv1d(MEL_BINS, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.enc_pos = nn.Parameter(torch.zeros(1, 1500, d))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        # norm_first layers leave the residual stream unnormalized; the stack needs a final norm
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.tok = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.tok.weight, std=d**-0.5)  # tied to the output, std 1 gives logits in the hundreds
        self.dec_pos = nn.Embedding(MAX_TEXT, d)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, VOCAB, bias=False)
        self.out.weight = self.tok.weight
        size = file_bytes(self)
        if size > FILE_BUDGET_BYTES:
            raise RuntimeError(f"model is {size} bytes, budget is {FILE_BUDGET_BYTES}")

    def encode(self, mel: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        """mel is (batch, 80, time). pad is True where the mel frame is padding."""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        t = x.size(1)
        x = x + self.enc_pos[:, :t]
        key_pad = None
        if pad is not None:
            key_pad = pad[:, ::2][:, :t]
        return self.encoder(x, src_key_padding_mask=key_pad)

    def decode(
        self,
        memory: torch.Tensor,
        tokens: torch.Tensor,
        mem_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens are teacher-forced ids (batch, text). Returns logits."""
        b, n = tokens.shape
        pos = torch.arange(n, device=tokens.device)
        y = self.tok(tokens) + self.dec_pos(pos)[None, :, :]
        causal = nn.Transformer.generate_square_subsequent_mask(n, device=tokens.device)
        h = self.decoder(y, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.out(h)

    @torch.no_grad()
    def generate(self, mel: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 128, beam: int = 5) -> torch.Tensor:
        """Length-normalized beam search. beam=1 is greedy."""
        memory = self.encode(mel)
        if beam <= 1 or mel.size(0) != 1:
            tokens = torch.full((mel.size(0), 1), bos_id, dtype=torch.long, device=mel.device)
            for _ in range(max_len - 1):
                logits = self.decode(memory, tokens)
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, nxt], dim=1)
                if (nxt == eos_id).all():
                    break
            return tokens
        beams = [(0.0, torch.tensor([[bos_id]], device=mel.device), False)]
        for _ in range(max_len - 1):
            pool = []
            for score, tokens, done in beams:
                if done:
                    pool.append((score, tokens, True))
                    continue
                logp = torch.log_softmax(self.decode(memory, tokens)[0, -1].float(), dim=-1)
                topv, topi = logp.topk(beam)
                for value, idx in zip(topv.tolist(), topi.tolist()):
                    nxt = torch.tensor([[idx]], device=mel.device)
                    pool.append((score + value, torch.cat([tokens, nxt], dim=1), idx == eos_id))
            pool.sort(key=lambda item: item[0] / max(1, item[1].size(1) - 1), reverse=True)
            beams = pool[:beam]
            if all(done for _, _, done in beams):
                break
        best = max(beams, key=lambda item: item[0] / max(1, item[1].size(1) - 1))
        return best[1]


def int8_megabytes(module: nn.Module | None = None) -> float:
    module = module or StudentASR()
    return file_bytes(module) / 1_000_000


def pack_int8(state: dict) -> dict:
    """Weight-only int8, one scale per output row. 1-D tensors stay float32."""
    packed = {}
    for name, w in state.items():
        if name == "out.weight":  # tied to tok.weight
            continue
        w = w.float()
        if w.dim() < 2:
            packed[name] = w
            continue
        scale = w.abs().flatten(1).amax(1).clamp(min=1e-8) / 127
        shape = (-1,) + (1,) * (w.dim() - 1)
        packed[name] = torch.round(w / scale.view(shape)).to(torch.int8)
        packed[name + ".scale"] = scale
    return packed


def unpack_int8(packed: dict) -> dict:
    state = {}
    for name, w in packed.items():
        if name.endswith(".scale"):
            continue
        if w.dtype == torch.int8:
            scale = packed[name + ".scale"]
            w = w.float() * scale.view((-1,) + (1,) * (w.dim() - 1))
        state[name] = w
    state["out.weight"] = state["tok.weight"]
    return state


def load_student(path) -> StudentASR:
    """Loads student.pt (float32) or student.int8.pt (packed)."""
    state = torch.load(path, map_location="cpu")
    if any(w.dtype == torch.int8 for w in state.values()):
        state = unpack_int8(state)
    model = StudentASR()
    if "decoder.norm.weight" not in state:  # first Kaggle run (Rslt/), trained without final norms
        model.encoder.norm = model.decoder.norm = None
    model.load_state_dict(state)
    return model.eval()

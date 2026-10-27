from __future__ import annotations

import utils.transport
import utils.session
import asyncio
import json
import math
import os
import subprocess
import sys
import threading
import time
import unicodedata
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "checkpoints" / "rslt3"
RESULTS_PATH = SITE_DIR / "results_public.json"
PAGE_PATH = SITE_DIR / "index.html"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
RATE_WINDOW_SECONDS = 60
RATE_LIMIT = 6

sys.path.insert(0, str(ROOT))
from model.bangla_tokenizer import BanglaSyllableTokenizer
from model.student import load_student, mel_frames


class PerIPRateLimiter:
    def __init__(self) -> None:
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, address: str) -> int | None:
        now = time.monotonic()
        cutoff = now - RATE_WINDOW_SECONDS
        with self.lock:
            events = self.requests[address]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= RATE_LIMIT:
                return max(1, math.ceil(RATE_WINDOW_SECONDS - (now - events[0])))
            events.append(now)
            if len(self.requests) > 10_000:
                for key in tuple(self.requests):
                    if not self.requests[key] or self.requests[key][-1] <= cutoff:
                        del self.requests[key]
        return None


rate_limiter = PerIPRateLimiter()
inference_lock = threading.Lock()


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-t",
                str(MAX_AUDIO_SECONDS),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=422, detail="Audio decoding timed out") from exc
    if decoded.returncode or not decoded.stdout:
        raise HTTPException(status_code=422, detail="Could not decode this audio file")
    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if not samples.size or not np.isfinite(samples).all():
        raise HTTPException(status_code=422, detail="Audio contains no usable samples")
    return samples


def run_transcription(samples: np.ndarray, model, extractor, tokenizer) -> str:
    features = extractor(samples, sampling_rate=16000, return_tensors="pt").input_features
    features = features[..., : mel_frames(len(samples))]
    with inference_lock, torch.inference_mode():
        ids = model.generate(
            features,
            bos_id=tokenizer.bos_id(),
            eos_id=tokenizer.eos_id(),
            max_len=128,
            beam=5,
        )[0].tolist()
    return unicodedata.normalize("NFC", tokenizer.decode(ids))


@asynccontextmanager
async def lifespan(app: FastAPI):
    required = ("student.pt", "bn.syllables.json", "preprocessor_config.json")
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"rslt3 model files missing: {', '.join(missing)}")
    torch.set_num_threads(1)
    app.state.extractor = WhisperFeatureExtractor.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    app.state.tokenizer = BanglaSyllableTokenizer.load(MODEL_DIR / "bn.syllables.json")
    app.state.model = load_student(MODEL_DIR / "student.pt")
    yield
    app.state.model = None
    app.state.extractor = None
    app.state.tokenizer = None


app = FastAPI(
    title="BanglaWhistle API",
    description="CPU transcription API for the rslt3 Bangla speech checkpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@app.get("/v1/results")
async def results() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, bool]:
    return {"ok": getattr(request.app.state, "model", None) is not None}


@app.post("/v1/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid content length") from exc
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="Invalid content length")
        if declared_length > MAX_UPLOAD_BYTES + 512 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds the 25 MB limit")
    address = request.headers.get("fly-client-ip") or (
        request.client.host if request.client else "unknown"
    )
    retry_after = rate_limiter.check(address)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached; try again shortly",
            headers={"Retry-After": str(retry_after)},
        )
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 25 MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Choose a non-empty audio file")
    samples = await asyncio.to_thread(decode_audio, content)
    text = await asyncio.to_thread(
        run_transcription,
        samples,
        request.app.state.model,
        request.app.state.extractor,
        request.app.state.tokenizer,
    )
    return {"text": text}

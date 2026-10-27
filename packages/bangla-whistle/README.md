# `bangla-whistle`

Small Node.js wrapper; it does not bundle the model or PyTorch.

```bash
npx bangla-whistle transcribe clip.wav
npx bangla-whistle transcribe clip.wav --remote
npx bangla-whistle transcribe clip.wav --url https://your-host.example
npx bangla-whistle serve
```

The local command requires `python3`, PyTorch, FFmpeg, and a BanglaWhistle source checkout. It looks for `edge-int8/` first, then the rslt3 checkpoint folder; set `BANGLA_WHISTLE_ROOT` and `BANGLA_WHISTLE_MODEL` to override discovery. If Python cannot import PyTorch, install it or use the Docker runtime. Remote mode needs `--url https://your-space.hf.space` or the `BANGLA_WHISTLE_URL` environment variable. Audio is limited to 25 MB by the API and only its first 30 seconds are transcribed.

`serve` starts the API locally if the checkout has Python, PyTorch, and Uvicorn; otherwise it starts the already-built Docker image `bangla-whistle`. Build it from `EDGE/BanglaWhistle` with `docker build -t bangla-whistle -f site/Dockerfile .`.

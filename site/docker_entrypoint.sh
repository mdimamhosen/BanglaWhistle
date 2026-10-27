#!/bin/sh
set -eu

if [ "${1:-}" = "transcribe" ]; then
  shift
  if [ "$#" -ne 1 ]; then
    echo "usage: docker run --rm -v \"\$PWD:/audio\" bangla-whistle transcribe /audio/clip.wav" >&2
    exit 2
  fi
  exec python /app/runtime/transcribe.py \
    --model /app/checkpoints/rslt3 \
    --audio "$1"
fi

exec uvicorn app:app --app-dir /app/site --host 0.0.0.0 --port 8080 --workers 1

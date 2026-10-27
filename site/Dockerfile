FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY site/requirements.txt /app/site/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && python -m pip install --no-cache-dir -r /app/site/requirements.txt

COPY model /app/model
COPY runtime/transcribe.py /app/runtime/transcribe.py
COPY site/app.py site/index.html site/results_public.json /app/site/
COPY site/docker_entrypoint.sh /app/site/docker_entrypoint.sh
COPY checkpoints/rslt3/student.pt /app/checkpoints/rslt3/student.pt
COPY checkpoints/rslt3/bn.syllables.json /app/checkpoints/rslt3/bn.syllables.json
COPY checkpoints/rslt3/preprocessor_config.json /app/checkpoints/rslt3/preprocessor_config.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
ENTRYPOINT ["/bin/sh", "/app/site/docker_entrypoint.sh"]

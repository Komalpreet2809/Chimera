# Chimera backend — FastAPI + PyTorch, built for Hugging Face Spaces (Docker SDK).
#
# Two deliberate choices:
#   1. CPU-only torch wheel — the full CUDA wheel is ~4GB and useless on a
#      free CPU Space. This keeps the image around 1GB.
#   2. GPT-2 weights are baked into the image at build time, so a cold start
#      doesn't re-download 500MB before serving its first request.

FROM python:3.11-slim

# HF Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    # Free Spaces get ~2 vCPUs. A naive 900-token forward pass takes seconds
    # there, so cap the live benchmark's longest sequence to keep it snappy.
    CHIMERA_BENCH_MAX_SEQ=512

WORKDIR /app

COPY requirements.deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.deploy.txt

COPY chimera ./chimera

# Pre-fetch the weights so they live in the image layer, not the cold start.
RUN mkdir -p "$HF_HOME" \
 && python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')" \
 && chown -R user:user /home/user /app

USER user
EXPOSE 7860

CMD ["uvicorn", "chimera.api.server:app", "--host", "0.0.0.0", "--port", "7860"]

# DXA QC service — CPU image (works on any x86_64 Linux host; GPU not required for inference).
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf TORCH_HOME=/tmp/torch \
    DXAQC_WEIGHTS=/app/weights DXAQC_DATA=/data DXAQC_DEVICE=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY dxaqc ./dxaqc
COPY scripts ./scripts
COPY weights ./weights

RUN useradd -m -u 10001 dxaqc && mkdir -p /data && chown dxaqc /data
USER dxaqc
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "dxaqc.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

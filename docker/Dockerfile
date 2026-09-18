FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ARG TORCH_INDEX=cpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DXA_MODEL=/app/models/radiance DXA_DEVICE=cpu \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY docker/requirements.txt /tmp/requirements.txt
COPY docker/requirements-cuda.txt /tmp/requirements-cuda.txt
RUN pip install --no-cache-dir --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    && if [ "$TORCH_INDEX" = cu126 ]; then pip install --no-cache-dir -r /tmp/requirements-cuda.txt; fi \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip check
COPY LICENSE /app/LICENSE
COPY src/__init__.py src/config.py /app/src/
COPY src/solution /app/src/solution
COPY models/radiance /app/models/radiance
COPY models/anatomy /app/models/anatomy
COPY src/utils/prepare_model.py /app/src/utils/prepare_model.py
COPY src/utils/prepare_segmentation.py /app/src/utils/prepare_segmentation.py
RUN python -m src.utils.prepare_model --verify-only \
    && if [ -f /app/models/anatomy/sam-vit-large.pt ]; then python -m src.utils.prepare_segmentation --verify-only; fi \
    && rm -rf /app/src/utils \
    && useradd --uid 10001 --create-home dxa
USER dxa
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD ["uvicorn", "src.solution.api:app", "--host", "0.0.0.0", "--port", "8080"]

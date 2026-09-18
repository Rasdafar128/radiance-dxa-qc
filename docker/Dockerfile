FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DXA_MODEL=/app/model DXA_DEVICE=cpu
WORKDIR /app
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -c /tmp/requirements.txt torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/__init__.py src/config.py /app/src/
COPY src/solution /app/src/solution
COPY artifacts/e0/final /app/model
RUN test -s /app/model/model.json && test -s /app/model/encoder.pt \
    && useradd --uid 10001 --create-home dxa
USER dxa
EXPOSE 8080
CMD ["uvicorn", "src.solution.api:app", "--host", "0.0.0.0", "--port", "8080"]

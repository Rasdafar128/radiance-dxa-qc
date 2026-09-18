FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip check && useradd --uid 10001 --create-home app
COPY LICENSE /app/LICENSE
COPY web /app/web
COPY src/__init__.py src/config.py /app/src/
COPY src/solution/__init__.py src/solution/dicom.py src/solution/segmentation.py /app/src/solution/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

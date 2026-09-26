"""Лёгкий веб-шлюз: временный файл → закрытый GPU API → результат в браузере."""

import asyncio
from contextlib import asynccontextmanager
import csv
import io
import lzma
import os
from pathlib import Path
import shutil
from tempfile import TemporaryFile
import zipfile
import zlib

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .previews import build_previews

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 256 * 1024 * 1024


def retry_member(source, target, index):
    """Повтор одного элемента ZIP по позиции, включая одинаковые имена."""
    source.seek(0)
    try:
        with zipfile.ZipFile(source, metadata_encoding="cp866") as original:
            members = [m for m in original.infolist() if not m.is_dir()]
            if index >= len(members) or len(members) > 10000:
                raise ValueError("Invalid member index")
            member = members[index]
            if member.file_size > 32 * 1024**2 or member.flag_bits & 1:
                raise ValueError("Unsupported member")
            with original.open(member) as stream, zipfile.ZipFile(target, "w") as archive:
                with archive.open(member.filename, "w") as output:
                    shutil.copyfileobj(stream, output)
        target.seek(0)
    except (zipfile.BadZipFile, ValueError, RuntimeError, OSError, zlib.error, lzma.LZMAError):
        raise HTTPException(400, "Не удалось повторить этот файл. Выберите исправный DICOM отдельно.") from None


@asynccontextmanager
async def lifespan(app):
    url = os.getenv("DXA_UPSTREAM_URL", "http://127.0.0.1:8080").rstrip("/")
    async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(600, connect=5), trust_env=False) as client:
        app.state.client = client
        app.state.lock = asyncio.Lock()
        yield


app = FastAPI(title="DXA Контроль · веб-шлюз", version="1.0", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def response_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    )
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/health")
async def upstream_health():
    try:
        response = await app.state.client.get("/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        if (result.get("status") != "ok" or result.get("model") != "Radiance"
                or result.get("version") != "1.0"
                or result.get("backbones") != ["dinov3-large", "medimageinsight"]):
            raise ValueError("Expected Radiance")
        return {"status": "ok", "model": "Radiance", "busy": app.state.lock.locked()}
    except (httpx.HTTPError, ValueError):
        raise HTTPException(503, "Сервис модели недоступен. Попробуйте позже.") from None


@app.post("/api/analyze")
async def analyze(request: Request, image_index: int | None = Query(default=None, ge=0, le=9999)):
    kind = request.headers.get("content-type", "").split(";")[0]
    if kind not in ("application/zip", "application/dicom"):
        raise HTTPException(415, "Выберите DICOM или ZIP с DICOM-файлами.")
    if app.state.lock.locked():
        raise HTTPException(503, "Сейчас обрабатывается другой пакет. Повторите проверку чуть позже.")
    async with app.state.lock:
        with TemporaryFile() as upload, TemporaryFile() as archive, TemporaryFile() as retry:
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Файл больше 256 МиБ. Разделите его на несколько архивов.")
                upload.write(chunk)
            if not size:
                raise HTTPException(400, "Файл пуст. Выберите другой файл.")
            upload.seek(0)
            if kind == "application/dicom":
                name = Path(request.query_params.get("filename", "image.dcm").replace("\\", "/")).name
                if not name or name in (".", "..") or "\x00" in name or len(name) > 240:
                    raise HTTPException(400, "Недопустимое имя файла. Переименуйте файл и повторите.")
                with zipfile.ZipFile(archive, "w") as bundle:
                    with bundle.open(name, "w") as member:
                        shutil.copyfileobj(upload, member)
                archive.seek(0)
                source = archive
            else:
                source = upload

            if image_index is not None:
                await run_in_threadpool(retry_member, source, retry, image_index)
                source = retry

            async def chunks():
                while block := source.read(1024 * 1024):
                    yield block

            try:
                await upstream_health()
                response = await app.state.client.post("/batch", content=chunks(),
                                                       headers={"Content-Type": "application/zip"})
                if response.status_code in (400, 413, 415):
                    raise HTTPException(response.status_code, "Не удалось прочитать архив. Проверьте формат и состав ZIP.")
                response.raise_for_status()
                if not response.headers.get("content-type", "").startswith("text/csv"):
                    raise ValueError("Unexpected upstream response")
                rows = list(csv.DictReader(io.StringIO(response.text)))
                if not rows or any("processing_status" not in row for row in rows):
                    raise ValueError("Invalid CSV")
                previews = await run_in_threadpool(build_previews, source, rows)
                return {"rows": rows, "csv": response.text, "model": "Radiance", "previews": previews}
            except httpx.TimeoutException:
                raise HTTPException(504, "Проверка не завершилась за 10 минут. Попробуйте меньший пакет.") from None
            except (httpx.HTTPError, ValueError):
                raise HTTPException(502, "Потеряно соединение с моделью. Повторите проверку позже.") from None


app.mount("/static", StaticFiles(directory=STATIC), name="static")

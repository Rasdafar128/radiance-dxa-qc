"""HTTP API DXA — ZIP в теле запроса, CSV в ответе, веса только с диска."""

from contextlib import asynccontextmanager
import os
from tempfile import TemporaryFile
from threading import Lock
import zipfile

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
import torch

from .. import config as C
from .batch import MAX_UPLOAD, predict_zip
from .model import Model


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.getenv("DXA_CPU_THREADS", "2")))
    app.state.model = Model.load(os.getenv("DXA_MODEL", str(C.ARTIFACTS / "e0" / "final")),
                                 os.getenv("DXA_DEVICE", "cpu"))
    app.state.lock = Lock()
    yield


app = FastAPI(title="DXA quality control", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "recipe": app.state.model.metadata["recipe"], "device": str(app.state.model.device)}


@app.post("/batch", responses={200: {"content": {"text/csv": {}}}})
async def batch(request: Request):
    if request.headers.get("content-type", "").split(";")[0] not in ("application/zip", "application/octet-stream"):
        raise HTTPException(415, "Send ZIP bytes with Content-Type: application/zip")
    with TemporaryFile() as upload:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "ZIP exceeds 256 MiB")
            upload.write(chunk)
        upload.seek(0)

        def process():
            # ponytail: один GPU-запрос за раз; очередь нужна при многопользовательской нагрузке.
            with app.state.lock:
                return predict_zip(app.state.model, upload).to_csv(index=False)

        try:
            csv = await run_in_threadpool(process)
        except (zipfile.BadZipFile, ValueError, NotImplementedError) as error:
            raise HTTPException(400, str(error)) from error
    return Response(csv, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="results.csv"'})

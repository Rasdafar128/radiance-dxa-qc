"""REST API + minimal web UI.

POST /api/v1/jobs              multipart: files (zip / dcm, several allowed) -> {job_id}
POST /api/v1/jobs/from-path    json: {"path": "/input"}  (folder mounted into the container)
GET  /api/v1/jobs/{id}         status, progress, rows when finished
GET  /api/v1/jobs/{id}/files/{results.csv|results.xlsx|overlays.zip|sr.zip}
GET  /api/v1/jobs/{id}/overlay/{study}/{image}.png
POST /api/v1/jobs/{id}/feedback  json: {"image_uid", "verdict": "agree"|"disagree", "comment"}
POST /api/v1/process           synchronous variant for small uploads -> rows
GET  /health
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, config
from .batch import _safe, run_batch
from .pipeline import Engine

log = logging.getLogger("dxaqc")
DATA = Path(os.environ.get("DXAQC_DATA", "/tmp/dxaqc"))
JOBS = DATA / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
ALLOWED_FILES = {"results.csv", "results.xlsx", "overlays.zip", "sr.zip"}

_engine: Engine | None = None
_engine_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # one model instance, sequential jobs
_state: dict[str, dict] = {}


def engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine(os.environ.get("DXAQC_WEIGHTS"), device=os.environ.get("DXAQC_DEVICE", "auto"))
        return _engine


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("DXAQC_LAZY") != "1":
        engine()  # load weights before accepting requests
    yield


app = FastAPI(title="DXA QC", version=__version__, lifespan=lifespan)


def _job_dir(job_id: str) -> Path:
    if not job_id.replace("-", "").isalnum():
        raise HTTPException(400, "bad job id")
    d = JOBS / job_id
    if not d.exists():
        raise HTTPException(404, "job not found")
    return d


def _run(job_id: str, input_path: Path):
    st = _state[job_id]
    st.update(status="running", started=time.time())
    try:
        df = run_batch(input_path, JOBS / job_id / "out", engine(),
                       progress=lambda i, n: st.update(done=i, total=n))
        st.update(status="finished", n_files=len(df), n_failures=int((df.processing_status != "Success").sum()))
        if len(df) == 0:
            st["warning"] = "DICOM-файлы не найдены"
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        st.update(status="failed", error=f"{type(e).__name__}: {e}")
    st["finished"] = time.time()
    (JOBS / job_id / "status.json").write_text(json.dumps(st, ensure_ascii=False))


async def _save_uploads(files: list[UploadFile], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or f"upload_{uuid.uuid4().hex}.dcm").name
        with open(dest / name, "wb") as out:
            shutil.copyfileobj(f.file, out)
    zips = list(dest.glob("*.zip"))
    return zips[0] if len(files) == 1 and zips else dest


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "model_loaded": _engine is not None}


@app.post("/api/v1/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    job_id = uuid.uuid4().hex
    inp = await _save_uploads(files, JOBS / job_id / "in")
    _state[job_id] = dict(job_id=job_id, status="queued", done=0, total=0, created=time.time())
    _pool.submit(_run, job_id, inp)
    return {"job_id": job_id}


class PathJob(BaseModel):
    path: str


@app.post("/api/v1/jobs/from-path")
def create_job_from_path(req: PathJob):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, f"path not found: {p}")
    job_id = uuid.uuid4().hex
    (JOBS / job_id).mkdir(parents=True)
    _state[job_id] = dict(job_id=job_id, status="queued", done=0, total=0, created=time.time())
    _pool.submit(_run, job_id, p)
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str, rows: bool = True):
    d = _job_dir(job_id)
    st = _state.get(job_id) or json.loads((d / "status.json").read_text())
    out = dict(st)
    csv = d / "out" / "results.csv"
    if rows and st.get("status") == "finished" and csv.exists():
        import pandas as pd
        rows_ = json.loads(pd.read_csv(csv, keep_default_na=False, dtype=str).to_json(orient="records", force_ascii=False))
        for r in rows_:
            r["overlay_url"] = (f"/api/v1/jobs/{job_id}/overlay/{_safe(r['study_uid'] or 'unknown')}/"
                                f"{_safe(r['image_uid'])}.png") if r.get("processing_status") == "Success" else ""
        out["rows"] = rows_
    return out


@app.get("/api/v1/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    if name not in ALLOWED_FILES:
        raise HTTPException(404)
    f = _job_dir(job_id) / "out" / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, filename=name)


@app.get("/api/v1/jobs/{job_id}/overlay/{study}/{image}")
def job_overlay(job_id: str, study: str, image: str):
    base = (_job_dir(job_id) / "out" / "overlays").resolve()
    f = (base / study / image).resolve()
    if not str(f).startswith(str(base)) or f.suffix != ".png" or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png")


class Feedback(BaseModel):
    image_uid: str
    verdict: str
    comment: str = ""


@app.post("/api/v1/jobs/{job_id}/feedback")
def feedback(job_id: str, fb: Feedback):
    d = _job_dir(job_id)
    with open(d / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(fb.model_dump(), ts=time.time()), ensure_ascii=False) + "\n")
    return {"ok": True}


@app.post("/api/v1/process")
async def process_sync(files: list[UploadFile] = File(...)):
    job_id = uuid.uuid4().hex
    inp = await _save_uploads(files, JOBS / job_id / "in")
    _state[job_id] = dict(job_id=job_id, status="queued", done=0, total=0, created=time.time())
    _pool.submit(_run, job_id, inp).result()
    return job_status(job_id)


@app.get("/api/v1/info")
def info():
    meta = _engine.qc.meta if _engine is not None else {}
    demo = os.environ.get("DXAQC_DEMO_PATH")
    return {"version": __version__, "model_version": meta.get("version"), "trained_at": meta.get("trained_at"),
            "cv": meta.get("cv", {}), "encoders": meta.get("encoders", config.EMBEDDERS),
            "demo_available": bool(demo and Path(demo).exists())}


@app.post("/api/v1/demo")
def demo_job():
    p = os.environ.get("DXAQC_DEMO_PATH")
    if not p or not Path(p).exists():
        raise HTTPException(404, "demo data is not configured (DXAQC_DEMO_PATH)")
    return create_job_from_path(PathJob(path=p))


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})

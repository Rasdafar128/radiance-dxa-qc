"""Batch processing of a folder / zip / single file with DICOM studies."""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import pandas as pd

from .dicom_io import extract_zip, iter_dicom_files, study_key
from .pipeline import Engine
from .report import to_frame, write_results
from .viz import render_overlay, save_png, write_secondary_capture, write_study_sr

log = logging.getLogger("dxaqc")


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)[:120]


def group_studies(root: Path) -> "OrderedDict[str, list[str]]":
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for p in iter_dicom_files(root):
        groups.setdefault(study_key(p), []).append(str(p))
    return groups


def run_batch(input_path: str | Path, output_dir: str | Path, engine: Engine, overlays: bool = True,
              sr: bool = True, progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        root = input_path
        if input_path.is_file() and zipfile.is_zipfile(input_path):
            tmp = Path(tempfile.mkdtemp(prefix="dxaqc_"))
            root = extract_zip(input_path, tmp)
        elif input_path.is_file():
            root = input_path.parent
        groups = group_studies(input_path if input_path.is_file() and tmp is None else root)
        log.info("found %d studies, %d files", len(groups), sum(map(len, groups.values())))
        rows = []
        ov_dir, sr_dir = output_dir / "overlays", output_dir / "sr"
        for i, (key, paths) in enumerate(groups.items(), 1):
            study_rows, records, results = engine.process_study(key, paths, root)
            rows.extend(study_rows)
            if overlays or sr:
                try:
                    items = [(r, results[r.pixel_hash]) for r in records
                             if r.pixels is not None and r.pixel_hash in results]
                    folder = _safe(records[0].study_uid or "unknown") if records else _safe(key)
                    if overlays:
                        rendered: dict[str, object] = {}
                        for rec, res in items:
                            stem = _safe(rec.sop_uid or Path(rec.path).stem)
                            first = rec.pixel_hash not in rendered
                            if first:
                                rendered[rec.pixel_hash] = render_overlay(rec.pixels, res)
                            rgb = rendered[rec.pixel_hash]
                            save_png(rgb, ov_dir / folder / f"{stem}.png")  # one PNG per file (UI lookup)
                            if first:  # one Secondary Capture per unique image
                                write_secondary_capture(rgb, rec.path, key, len(rendered), " | ".join(res.details),
                                                        ov_dir / folder / f"{stem}_sc.dcm")
                    if sr:
                        write_study_sr(key, items, sr_dir / f"{folder}_sr.dcm")
                except Exception as e:  # noqa: BLE001 - visualisation must not break the table
                    log.error("visualisation failed for %s: %s", key, e)
            if progress:
                progress(i, len(groups))
        df = to_frame(rows)
        write_results(df, output_dir)
        for sub in ("overlays", "sr"):
            d = output_dir / sub
            if d.exists():
                with zipfile.ZipFile(output_dir / f"{sub}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                    for f in sorted(d.rglob("*")):
                        if f.is_file():
                            z.write(f, f.relative_to(d))
        return df
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

"""Robust DICOM discovery and reading.

Every public function here must never raise on a single bad file: errors are
captured in ``ImageRecord.error`` so batch processing can report them.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

SKIP_NAMES = {"DICOMDIR", ".DS_Store"}


@dataclass
class ImageRecord:
    path: str
    study_uid: str = ""
    series_uid: str = ""
    sop_uid: str = ""
    instance_number: int | None = None
    rows: int = 0
    cols: int = 0
    pixels: np.ndarray | None = None  # uint8, 2-D
    pixel_hash: str = ""
    meta: dict = field(default_factory=dict)
    error: str | None = None


def _looks_like_dicom(p: Path) -> bool:
    if p.name in SKIP_NAMES or p.name.startswith("._"):
        return False
    if p.suffix.lower() in {".dcm", ".dicom"}:
        return True
    if p.suffix == "":
        try:
            with open(p, "rb") as f:
                head = f.read(132)
            return len(head) == 132 and head[128:132] == b"DICM"
        except OSError:
            return False
    return False


def iter_dicom_files(root: str | os.PathLike) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        if _looks_like_dicom(root):
            yield root
        return
    for dirpath, _, files in os.walk(root, followlinks=True):  # test sets are often mounted via symlinks
        for name in sorted(files):
            p = Path(dirpath) / name
            if _looks_like_dicom(p):
                yield p


def extract_zip(zip_path: str | os.PathLike, out_dir: str | os.PathLike) -> Path:
    """Extract a zip, fixing cp866 names produced by Windows archivers."""
    out_dir = Path(out_dir)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("cp866")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            target = (out_dir / name).resolve()
            if not target.is_relative_to(out_dir.resolve()):
                raise ValueError('Archive path escapes output directory')
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return out_dir


def to_uint8(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):  # RGB(A) secondary capture
        a = a[..., :3].mean(-1)
    elif a.ndim == 3:  # multi-frame: take first frame
        a = a[0]
    src_dtype = a.dtype
    a = a.astype(np.float32)
    if src_dtype != np.uint8:
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        a = a * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    bits = int(getattr(ds, "BitsStored", 8) or 8)
    if bits > 8 or a.max() > 255 or a.min() < 0:
        lo, hi = np.percentile(a, [0.5, 99.5])
        a = (a - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def read_image(path: str | os.PathLike) -> ImageRecord:
    rec = ImageRecord(path=str(path))
    try:
        ds = pydicom.dcmread(path, force=True)
        rec.study_uid = str(ds.get("StudyInstanceUID", "") or "")
        rec.series_uid = str(ds.get("SeriesInstanceUID", "") or "")
        rec.sop_uid = str(ds.get("SOPInstanceUID", "") or "")
        try:
            rec.instance_number = int(ds.get("InstanceNumber"))
        except (TypeError, ValueError):
            rec.instance_number = None
        rec.meta = {
            "Modality": str(ds.get("Modality", "")),
            "Manufacturer": str(ds.get("Manufacturer", "")),
            "Model": str(ds.get("ManufacturerModelName", "")),
            "SoftwareVersions": str(ds.get("SoftwareVersions", "")),
            "SeriesDescription": str(ds.get("SeriesDescription", "")),
            "SOPClassUID": str(ds.get("SOPClassUID", "")),
        }
        if "PixelData" not in ds:
            rec.error = "no PixelData (not an image object)"
            return rec
        px = to_uint8(ds.pixel_array, ds)
        if px.ndim != 2 or min(px.shape) < 32:
            rec.error = f"unsupported image shape {px.shape}"
            return rec
        rec.pixels = px
        rec.rows, rec.cols = px.shape
        rec.pixel_hash = hashlib.md5(px.tobytes() + str(px.shape).encode()).hexdigest()
    except InvalidDicomError as e:
        rec.error = f"invalid DICOM: {e}"
    except Exception as e:  # noqa: BLE001 - must never crash the batch
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def study_key(path: str | os.PathLike) -> str:
    """StudyInstanceUID from the header (fast), falling back to the parent directory."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=["StudyInstanceUID"])
        uid = str(ds.get("StudyInstanceUID", "") or "").strip()
        if uid:
            return uid
    except Exception:  # noqa: BLE001
        pass
    return f"dir:{Path(path).parent}"

"""Полный кадр DICOM — одинаковая предобработка для обучения и сервиса."""

import numpy as np
from PIL import Image, ImageOps


PREPROCESS = {
    "size": 224, "resize": "bicubic_letterbox", "intensity": "stored_bit_range",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "aspect": "pixel_grid", "padding": 0,
}


def pixels(ds):
    """Один монохромный кадр; неизвестные форматы не превращаем в норму."""
    photo = str(ds.get("PhotometricInterpretation", ""))
    if photo not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"Unsupported photometric interpretation: {photo}")
    if int(ds.get("NumberOfFrames", 1)) != 1:
        raise ValueError("Only single-frame DICOM is supported")
    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise ValueError("Only grayscale DICOM is supported")
    if int(ds.Rows) * int(ds.Columns) > 16_000_000:
        raise ValueError("Image exceeds 16 million pixels")
    a = ds.pixel_array
    if a.ndim != 2 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError("Invalid pixel dimensions or values")
    bits = int(ds.BitsStored)
    if not 1 <= bits <= 16:
        raise ValueError("Only 1–16 stored bits are supported")
    low = -(2 ** (bits - 1)) if int(ds.PixelRepresentation) else 0
    high = low + 2**bits - 1
    if a.min() < low or a.max() > high or np.ptp(a.astype(float)) == 0:
        raise ValueError("Constant image or values outside the stored bit range")
    # Диапазон всего кадра не растягиваем: сохраняем контраст мелких предметов.
    a = (a.astype(np.float32) - low) / (high - low)
    if photo == "MONOCHROME1":
        a = 1 - a
    return np.rint(a * 255).astype(np.uint8)


def prepare(ds, recipe=PREPROCESS):
    image = Image.fromarray(pixels(ds)).convert("RGB")
    image = ImageOps.pad(image, (recipe["size"], recipe["size"]),
                         method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    a = np.asarray(image, dtype=np.float32) / 255
    a = (a - np.asarray(recipe["mean"], dtype=np.float32)) / np.asarray(recipe["std"], dtype=np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1))

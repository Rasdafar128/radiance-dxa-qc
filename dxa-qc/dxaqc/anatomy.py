"""Region and laterality detection from pixels only (no reliance on tags or file names)."""
from __future__ import annotations

import cv2
import numpy as np

SPINE = "Поясничный отдел позвоночника"
HIP = "Проксимальный отдел бедра"


def bone_mask(img: np.ndarray) -> np.ndarray:
    thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.GaussianBlur(img, (5, 5), 0) > thr


def hip_side_score(img: np.ndarray) -> float:
    """> 0: pelvis on image right => patient's right hip (radiological convention).

    The femoral shaft is found in the bottom 20% of the bone mask; bone mass in the
    upper 45% is compared on both sides of it (pelvis/acetabulum vs greater trochanter).
    """
    a = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    h, w = a.shape
    m = bone_mask(img).astype(np.float32)
    cols = m[int(h * 0.8):].sum(0)
    xsh = int((cols * np.arange(w)).sum() / max(cols.sum(), 1.0))
    top = (a * m)[: int(h * 0.45)]
    left, right = top[:, :xsh].sum(), top[:, xsh:].sum()
    return float((right - left) / (right + left + 1e-6))


def canonical_hip(img: np.ndarray, side: str) -> np.ndarray:
    """Mirror left hips so every hip looks like a right hip (trochanter on image left)."""
    return img if side == "R" else np.ascontiguousarray(img[:, ::-1])

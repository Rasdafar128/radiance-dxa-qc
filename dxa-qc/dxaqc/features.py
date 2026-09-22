"""Feature extraction shared by training and inference.

For every unique image we embed the raw and the horizontally flipped picture with every backbone:
* region model   -> dinov2_b(raw)            (trained on raw+flipped => orientation invariant)
* spine tasks    -> (emb(raw) + emb(flip)) / 2  (axis tilt sign is irrelevant)
* hip tasks      -> emb(canonical) = raw for right hips, flip for left hips
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .anatomy import canonical_hip, hip_side_score
from .embed import Embedder
from .geometry import hip_geometry, spine_geometry


class FeatureExtractor:
    def __init__(self, embedders=None, device: str | None = None, weights_dir=None):
        names = list(embedders or config.EMBEDDERS)
        wdir = weights_dir if weights_dir is not None else config.WEIGHTS_DIR
        self.embedders = {n: Embedder(n, device=device, weights_dir=wdir) for n in names}

    def embed_raw_and_flip(self, imgs: list[np.ndarray]) -> tuple[dict, dict]:
        flipped = [np.ascontiguousarray(im[:, ::-1]) for im in imgs]
        raw = {n: e(imgs) for n, e in self.embedders.items()}
        flp = {n: e(flipped) for n, e in self.embedders.items()}
        return raw, flp


def task_embeddings(raw: dict, flp: dict, regions, sides) -> dict[str, np.ndarray]:
    regions, sides = np.asarray(regions), np.asarray(sides)
    out = {}
    for n in raw:
        spine = (regions == "spine")[:, None]
        left = (sides == "L")[:, None]
        out[n] = np.where(spine, (raw[n] + flp[n]) / 2, np.where(left, flp[n], raw[n]))
    return out


def geometry_frame(imgs, regions, sides) -> tuple[pd.DataFrame, list[dict]]:
    rows, viz = [], []
    for im, reg, sd in zip(imgs, regions, sides):
        g = spine_geometry(im) if reg == "spine" else hip_geometry(canonical_hip(im, sd))
        viz.append(g.pop("_viz", {}))
        rows.append(g)
    return pd.DataFrame(rows), viz


def hip_sides(imgs, regions) -> list[str]:
    return ["" if r != "hip" else ("R" if hip_side_score(im) > 0 else "L") for im, r in zip(imgs, regions)]


def compute_task_embeddings(fx: FeatureExtractor, imgs, regions, sides, raw_cache: dict | None = None) -> dict:
    """Inference-time equivalent of ``task_embeddings`` that runs only the passes actually needed:
    raw for spines and right hips, mirrored for spines and left hips; reuses cached raw embeddings."""
    regions, sides = np.asarray(regions), np.asarray(sides)
    raw_cache = raw_cache or {}
    n = len(imgs)
    need_raw = np.where((regions == "spine") | (sides == "R"))[0]
    need_flip = np.where((regions == "spine") | (sides == "L"))[0]
    spine, left = (regions == "spine")[:, None], (sides == "L")[:, None]
    out = {}
    for name in config.EMBEDDERS:
        e = fx.embedders[name]
        part_raw = None if name in raw_cache or len(need_raw) == 0 else e([imgs[i] for i in need_raw])
        part_flip = e([np.ascontiguousarray(imgs[i][:, ::-1]) for i in need_flip]) if len(need_flip) else None
        dim = next(x.shape[1] for x in (raw_cache.get(name), part_raw, part_flip) if x is not None)
        R = np.zeros((n, dim), np.float32)
        Fl = np.zeros((n, dim), np.float32)
        if name in raw_cache:
            R[:] = raw_cache[name]
        elif part_raw is not None:
            R[need_raw] = part_raw
        if part_flip is not None:
            Fl[need_flip] = part_flip
        out[name] = np.where(spine, (R + Fl) / 2, np.where(left, Fl, R))
    return out


def available_cpus() -> int:
    """CPU count respecting the container cgroup quota (os.cpu_count() reports host cores)."""
    import os
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1

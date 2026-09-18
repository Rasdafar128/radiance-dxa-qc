"""Anatomical region classifier on frozen embeddings + a simple out-of-distribution check."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMBEDDER = "dinov2_b"


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


@dataclass
class RegionClassifier:
    clf: object = None
    refs: np.ndarray | None = None  # L2-normalised training embeddings (raw + flipped)
    ood_threshold: float = 1.0
    classes: tuple = ("hip", "spine")

    def fit(self, emb_raw: np.ndarray, emb_flip: np.ndarray, regions: np.ndarray, image_ids: np.ndarray):
        X = np.vstack([emb_raw, emb_flip])
        y = np.r_[regions, regions]
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(X, y)
        self.classes = tuple(self.clf.classes_)
        self.refs = _l2(X)
        ids = np.r_[image_ids, image_ids]
        sims = self.refs @ self.refs.T
        sims[ids[:, None] == ids[None, :]] = -1  # exclude the image itself and its mirror
        nn_dist = 1 - sims.max(1)
        self.ood_threshold = float(np.quantile(nn_dist, 0.99) * 1.5)
        return self

    def predict(self, emb_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        proba = self.clf.predict_proba(emb_raw)
        region = np.asarray(self.classes)[proba.argmax(1)]
        nn_dist = 1 - (_l2(emb_raw) @ self.refs.T).max(1)
        return region, proba.max(1), nn_dist > self.ood_threshold

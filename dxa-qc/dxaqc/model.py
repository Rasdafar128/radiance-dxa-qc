"""Spec-driven ensemble: per task, mean of logits of small L2-logistic regressions.

The same code is used for cross-validation, final training and inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _lr(C: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def block_matrix(item, geom: pd.DataFrame, emb: dict[str, np.ndarray]) -> np.ndarray:
    kind, what, _ = item
    if kind == "geom":
        return geom.reindex(columns=what).to_numpy(dtype=float)
    return emb[what]


def contralateral(p: np.ndarray, studies, sides) -> np.ndarray:
    """Max probability over hips of the opposite side in the same study (nan if none)."""
    df = pd.DataFrame({"p": p, "st": studies, "sd": sides})
    best = df.groupby(["st", "sd"]).p.max().to_dict()
    return np.array([best.get((st, "L" if sd == "R" else "R"), np.nan) for st, sd in zip(df.st, df.sd)])


@dataclass
class RegionModel:
    region: str
    spec: dict = field(default_factory=lambda: config.SPEC)
    beta: dict = field(default_factory=lambda: config.CONTEXT_BETA)
    any_spec: dict = field(default_factory=lambda: config.ANY_SPEC)
    any_blend: float = config.ANY_BLEND
    models: dict = field(default_factory=dict)  # task -> list of fitted pipelines (or constant)
    any_models: list = field(default_factory=list)

    @property
    def tasks(self):
        return config.TASKS[self.region]

    def fit(self, geom, emb, Y: pd.DataFrame):
        for t in self.tasks:
            y = Y[f"y_{t}"].to_numpy(int)
            fitted = []
            for item in self.spec[t]:
                X = block_matrix(item, geom, emb)
                fitted.append(float(y.mean()) if y.min() == y.max() else _lr(item[2]).fit(X, y))
            self.models[t] = fitted
        if self.any_blend:
            y = Y["y_any"].to_numpy(int)
            self.any_models = [float(y.mean()) if y.min() == y.max() else _lr(item[2]).fit(
                block_matrix(item, geom, emb), y) for item in self.any_spec[self.region]]
        return self

    def predict_tasks(self, geom, emb, studies=None, sides=None) -> pd.DataFrame:
        out = {}
        for t in self.tasks:
            logits = []
            for item, mdl in zip(self.spec[t], self.models[t]):
                X = block_matrix(item, geom, emb)
                p = np.full(len(X), mdl) if isinstance(mdl, float) else mdl.predict_proba(X)[:, 1]
                logits.append(logit(p))
            out[t] = sigmoid(np.mean(logits, 0))
        if self.region == "hip" and studies is not None:
            base = {t: out[t].copy() for t in self.tasks}
            for t in self.tasks:
                b = self.beta.get(t, 0.0)
                if b:
                    o = contralateral(base[t], studies, sides)
                    out[t] = sigmoid(logit(base[t]) + b * np.nan_to_num(logit(o), nan=0.0))
        P = pd.DataFrame(out)
        P["any"] = 1 - np.prod([1 - P[t].to_numpy() for t in self.tasks], 0)
        if self.any_blend and self.any_models:
            logits = []
            for item, mdl in zip(self.any_spec[self.region], self.any_models):
                X = block_matrix(item, geom, emb)
                p = np.full(len(X), mdl) if isinstance(mdl, float) else mdl.predict_proba(X)[:, 1]
                logits.append(logit(p))
            direct = np.mean(logits, 0)
            w = self.any_blend
            P["any"] = sigmoid((1 - w) * logit(P["any"].to_numpy()) + w * direct)
        return P


def cross_val_oof(region, geom, emb, Y, studies, sides, reps=5, seed=0, folds=5, spec=None,
                  beta=None, any_spec=None, any_blend=None) -> list[pd.DataFrame]:
    """Repeated study-grouped K-fold OOF task probabilities."""
    y_any = Y["y_any"].to_numpy(int)
    res = []
    for r in range(reps):
        P = pd.DataFrame(index=range(len(Y)), columns=config.TASKS[region] + ["any"], dtype=float)
        for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed + r).split(Y, y_any, studies):
            kw = {k: v for k, v in (("spec", spec), ("beta", beta), ("any_spec", any_spec),
                                    ("any_blend", any_blend)) if v is not None}
            m = RegionModel(region, **kw).fit(geom.iloc[tr], {k: v[tr] for k, v in emb.items()}, Y.iloc[tr])
            p = m.predict_tasks(geom.iloc[te], {k: v[te] for k, v in emb.items()},
                                np.asarray(studies)[te], np.asarray(sides)[te])
            P.iloc[te] = p.to_numpy()
        res.append(P)
    return res


def best_f1_threshold(y, p) -> tuple[float, float]:
    cands = np.unique(np.round(p, 4))
    scores = [(f1_score(y, p >= c), c) for c in cands]
    f1, th = max(scores)
    return float(th), float(f1)


class QualityModel:
    """Both regions + decision thresholds; persisted with joblib."""

    def __init__(self, regions: dict[str, RegionModel] | None = None, thresholds: dict | None = None,
                 meta: dict | None = None):
        self.regions = regions or {}
        self.thresholds = thresholds or {}
        self.meta = meta or {}

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"regions": self.regions, "thresholds": self.thresholds, "meta": self.meta}, path)
        path.with_suffix(".json").write_text(json.dumps({"thresholds": self.thresholds, "meta": self.meta},
                                                        ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path):
        obj = joblib.load(path)
        return cls(obj["regions"], obj["thresholds"], obj["meta"])

    def calibrated_any(self, region: str, p_any_raw) -> np.ndarray:
        a, b = self.thresholds[region].get("calib", (1.0, 0.0))
        return sigmoid(a * logit(p_any_raw) + b)

    def decide(self, region: str, probs, blocked: set | None = None) -> tuple[int, list[str], float]:
        """quality_class from calibrated P(any violation); violation types from per-task thresholds.
        ``blocked``: tasks ruled out by a measurement (e.g. axis tilt below AXIS_MIN_ANGLE_DEG)."""
        th = self.thresholds[region]
        p_any = float(self.calibrated_any(region, probs["any"]))
        qc = int(p_any >= th["any"])
        viol = []
        if qc:
            ratios = {t: probs[t] / max(th[t], 1e-6)
                      for t in config.TASKS[region] if not blocked or t not in blocked}
            viol = [t for t, r in ratios.items() if r >= 1.0]
            if not viol and ratios:
                viol = [max(ratios, key=ratios.get)]
            qc = int(bool(viol))
        names = []
        for t in viol:
            n = config.VIOLATION_NAMES[(region, t)]
            if n not in names:
                names.append(n)
        return qc, names, p_any

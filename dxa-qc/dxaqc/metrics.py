"""Diagnostic metrics with study-level (cluster) bootstrap 95% confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_metrics(y, p, th) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    yhat = (p >= th).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    two = len(np.unique(y)) == 2
    return dict(n=len(y), n_pos=int(y.sum()), sensitivity=sens, specificity=spec,
                balanced_accuracy=np.nanmean([sens, spec]), f1=f1_score(y, yhat, zero_division=0),
                roc_auc=roc_auc_score(y, p) if two else np.nan,
                pr_auc=average_precision_score(y, p) if two else np.nan, threshold=th,
                tp=tp, fp=fp, tn=tn, fn=fn)


def macro_f1(Y: np.ndarray, Yhat: np.ndarray) -> float:
    return float(np.mean([f1_score(Y[:, j], Yhat[:, j], zero_division=0) for j in range(Y.shape[1])]))


def cluster_bootstrap(fn, groups, n_boot=2000, seed=0) -> tuple[float, float]:
    """fn(index_array) -> metric; resample studies with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        try:
            v = fn(idx)
        except ValueError:
            continue
        if v == v:
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

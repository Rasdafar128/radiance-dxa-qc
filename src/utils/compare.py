"""Парный bootstrap разности метрик на одних и тех же исследованиях.

    python -m src.utils.compare artifacts/e0 artifacts/e2-dinov2-large
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import config as C


def paired(baseline, candidate, resamples=2000):
    frames = [f.sort_values("image_id").reset_index(drop=True) for f in (baseline, candidate)]
    identity = ["image_id", "study", "fold", "anatomical_region", "true_quality",
                *[f"true_{t}" for t in C.TARGETS]]
    pd.testing.assert_frame_equal(frames[0][identity], frames[1][identity])
    if frames[0].image_id.duplicated().any():
        raise ValueError("Repeated OOF image")
    y = frames[0][["true_quality", *[f"true_{t}" for t in C.TARGETS]]].to_numpy()
    decisions = [f[["pred_quality", *[f"pred_{t}" for t in C.TARGETS]]].to_numpy().astype(bool) for f in frames]
    probabilities = [f.prob_quality.to_numpy() for f in frames]

    def score(which, ix):
        truth, pred = y[ix] == 1, decisions[which][ix]
        known = np.isfinite(y[ix])
        tp = (known & truth & pred).sum(axis=0)
        denominator = (known & truth).sum(axis=0) + (known & pred).sum(axis=0)
        f1 = np.divide(2 * tp, denominator, out=np.zeros(len(C.TARGETS) + 1), where=denominator > 0)
        auc = roc_auc_score(truth[:, 0], probabilities[which][ix]) if np.unique(truth[:, 0]).size == 2 else np.nan
        return np.array([f1[0], auc, f1[1:].mean(), *f1[1:]])

    groups = [np.flatnonzero(frames[0].study.to_numpy() == g) for g in sorted(frames[0].study.unique())]
    all_indices = np.arange(len(y))
    before, after = score(0, all_indices), score(1, all_indices)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(resamples):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        samples.append(score(1, ix) - score(0, ix))
    samples = np.array(samples).reshape(resamples, len(before))
    result = {}
    for i, name in enumerate(["quality_f1", "quality_roc_auc", "macro_f1", *C.TARGETS]):
        valid = samples[:, i][np.isfinite(samples[:, i])]
        result[name] = dict(baseline=float(before[i]) if np.isfinite(before[i]) else None,
                            candidate=float(after[i]) if np.isfinite(after[i]) else None,
                            delta=float(after[i] - before[i]) if np.isfinite(after[i] - before[i]) else None,
                            ci95=np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
                            valid_resamples=len(valid))
    return dict(procedure="paired-study-bootstrap-fixed-oof-v1", seed=42, resamples=resamples,
                images=len(y), studies=len(groups), metrics=result,
                limitation="Development OOF; intervals exclude training and model-selection variability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = [pd.read_csv(p / "oof.csv") for p in (args.baseline, args.candidate)]
    # Равные прогнозы при обратном порядке должны давать нулевую разность.
    same = paired(baseline, baseline.iloc[::-1], resamples=10)
    assert all((m["delta"] == 0 and m["ci95"] == [0, 0]) or
               (m["delta"] is None and m["ci95"] is None) for m in same["metrics"].values())
    result = paired(baseline, candidate)
    result.update(baseline=str(args.baseline), candidate=str(args.candidate))
    destination = args.candidate / f"versus_{args.baseline.name}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))

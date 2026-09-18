"""Tune the direct «есть нарушение» head that is mixed into the noisy-OR.

The noisy-OR over criteria answers *which* rule is broken; a head trained straight on the
study-level verdict also catches images the expert failed without ticking a criterion.
This script sweeps the encoder recipe of that head and the logit mixing weight, and reports
the pooled image-level ROC-AUC / F1 — the two numbers the organizers score.

python experiments/any_blend.py --reps 5
Needs artifacts/features_all.pkl written by experiments/extract_all.py.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.features import task_embeddings  # noqa: E402
from dxaqc.model import best_f1_threshold, cross_val_oof, logit, sigmoid  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402


def pooled(oof_by_region, reps):
    """Per-region Platt calibration on OOF, then pooled ROC-AUC and best-F1 over all images."""
    y_all, p_all = [], []
    for y, oofs in oof_by_region:
        yc = np.concatenate([y] * reps)
        pc = np.concatenate([o["any"].values.astype(float) for o in oofs])
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc)[:, None], yc)
        avg = np.mean([o["any"].values.astype(float) for o in oofs], 0)
        y_all.append(y)
        p_all.append(sigmoid(cal.coef_[0, 0] * logit(avg) + cal.intercept_[0]))
    y, p = np.concatenate(y_all), np.concatenate(p_all)
    th, f1 = best_f1_threshold(y, p)
    return float(roc_auc_score(y, p)), float(f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--features", default="artifacts/features_all.pkl")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-emb", type=int, default=2)
    ap.add_argument("--weights", type=float, nargs="*", default=[0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
    a = ap.parse_args()

    u = build_dataset(Path(a.data))
    with open(a.features, "rb") as f:
        cache = pickle.load(f)
    geom, names = cache["geom"], sorted(cache["raw"])
    temb = task_embeddings(cache["raw"], cache["flp"], u.region.values, u.side.values)
    print("encoders:", names, flush=True)

    per_region = {}
    for reg in ("spine", "hip"):
        idx = np.where(((u.region == reg) & u.labeled).values)[0]
        sub = u.iloc[idx].reset_index(drop=True)
        per_region[reg] = dict(
            sub=sub, g=geom.iloc[idx].reset_index(drop=True),
            e={k: v[idx] for k, v in temb.items()},
            Y=sub[[f"y_{t}" for t in config.TASKS[reg]] + ["y_any"]].astype(int))

    combos = [c for k in range(1, a.max_emb + 1) for c in itertools.combinations(names, k)]
    rows = []
    for combo in combos:
        any_spec = {r: [("emb", n, 0.03) for n in combo] for r in ("spine", "hip")}
        for w in a.weights:
            if w == 0 and combo != combos[0]:
                continue  # the pure noisy-OR does not depend on the head recipe
            oof_by_region = []
            for reg in ("spine", "hip"):
                d = per_region[reg]
                oofs = cross_val_oof(reg, d["g"], d["e"], d["Y"], d["sub"].folder.values, d["sub"].side.values,
                                     reps=a.reps, seed=config.SEED, any_spec=any_spec, any_blend=w)
                oof_by_region.append((d["Y"]["y_any"].to_numpy(int), oofs))
            auc, f1 = pooled(oof_by_region, a.reps)
            rows.append(dict(head="+".join(combo), w=w, roc_auc=round(auc, 4), f1=round(f1, 4)))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    Path("reports").mkdir(exist_ok=True)
    R.to_csv("reports/any_blend.csv", index=False)
    print("\n== best by ROC-AUC\n", R.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

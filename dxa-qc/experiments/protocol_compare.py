"""Evaluate our model with the teammate's protocol so the numbers are directly comparable.

Outer: 3 study-grouped folds (stratified by quality_class), pooled OOF over all images.
Inner: thresholds and calibration fitted only inside each outer train.
Reports: F1 of quality_class, ROC-AUC of quality_prob, macro-F1 over the five
(region, criterion) pairs — the definition used in their table — and over the four
official violation names, plus per-criterion F1.

python experiments/protocol_compare.py --seeds 42 137 [--features artifacts/features_all.pkl]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.features import task_embeddings  # noqa: E402
from dxaqc.model import (QualityModel, RegionModel, best_f1_threshold, cross_val_oof, logit,  # noqa: E402
                         sigmoid)
from dxaqc.pipeline import blocked_tasks  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402


def evaluate(u, geom, temb, seed, outer=3, inner=2, spec=None, beta=None, use_rule=True):
    records = []
    for reg in ("spine", "hip"):
        idx = np.where(((u.region == reg) & u.labeled).values)[0]
        sub = u.iloc[idx].reset_index(drop=True)
        g = geom.iloc[idx].reset_index(drop=True)
        e = {k: v[idx] for k, v in temb.items()}
        Y = sub[[f"y_{t}" for t in config.TASKS[reg]] + ["y_any"]].astype(int)
        groups, sides = sub.folder.values, sub.side.values
        for tr, te in StratifiedGroupKFold(outer, shuffle=True, random_state=seed).split(sub, Y["y_any"], groups):
            oofs = cross_val_oof(reg, g.iloc[tr].reset_index(drop=True), {k: v[tr] for k, v in e.items()},
                                 Y.iloc[tr].reset_index(drop=True), groups[tr], sides[tr],
                                 reps=1, seed=seed, folds=inner, spec=spec, beta=beta)
            th = {}
            for t in config.TASKS[reg]:
                th[t], _ = best_f1_threshold(Y[f"y_{t}"].values[tr], oofs[0][t].values.astype(float))
            pc = oofs[0]["any"].values.astype(float)
            cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc)[:, None], Y["y_any"].values[tr])
            th["calib"] = (float(cal.coef_[0, 0]), float(cal.intercept_[0]))
            th["any"], _ = best_f1_threshold(Y["y_any"].values[tr], sigmoid(th["calib"][0] * logit(pc) + th["calib"][1]))
            kw = {k: v for k, v in (("spec", spec), ("beta", beta)) if v is not None}
            model = RegionModel(reg, **kw).fit(g.iloc[tr], {k: v[tr] for k, v in e.items()}, Y.iloc[tr])
            P = model.predict_tasks(g.iloc[te].reset_index(drop=True), {k: v[te] for k, v in e.items()},
                                    groups[te], sides[te])
            qm = QualityModel(thresholds={reg: th})
            for j, i in enumerate(te):
                probs = {t: float(P.iloc[j][t]) for t in config.TASKS[reg]}
                probs["any"] = float(P.iloc[j]["any"])
                gi = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in g.iloc[i].items()}
                blocked = blocked_tasks(reg, gi) if use_rule else set()
                qc, names, p_any = qm.decide(reg, probs, blocked)
                records.append(dict(region=reg, folder=groups[i], y=int(Y["y_any"].values[i]), p=p_any, pred=qc,
                                    gt_names=[config.VIOLATION_NAMES[(reg, t)] for t in config.TASKS[reg]
                                              if Y[f"y_{t}"].values[i] == 1],
                                    pred_names=names,
                                    gt_pairs=[t for t in config.TASKS[reg] if Y[f"y_{t}"].values[i] == 1],
                                    pred_pairs=[t for t in config.TASKS[reg]
                                                if config.VIOLATION_NAMES[(reg, t)] in names]))
    R = pd.DataFrame(records)
    y, p, yhat = R.y.values, R.p.values, R.pred.values
    pair_f1, name_f1 = {}, {}
    for reg in ("spine", "hip"):
        m = R.region == reg
        for t in config.TASKS[reg]:
            pair_f1[f"{reg}/{t}"] = f1_score([t in v for v in R[m].gt_pairs], [t in v for v in R[m].pred_pairs],
                                             zero_division=0)
    for nm in sorted(set(config.VIOLATION_NAMES.values())):
        name_f1[nm] = f1_score([nm in v for v in R.gt_names], [nm in v for v in R.pred_names], zero_division=0)
    return dict(f1=f1_score(y, yhat), roc_auc=roc_auc_score(y, p),
                macro_f1_pairs=float(np.mean(list(pair_f1.values()))),
                macro_f1_names=float(np.mean(list(name_f1.values())))), pair_f1, name_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--features", default="artifacts/features_all.pkl")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 137])
    ap.add_argument("--no-rule", action="store_true", help="disable the axis measurement rule")
    a = ap.parse_args()
    u = build_dataset(Path(a.data))
    with open(a.features, "rb") as f:
        cache = pickle.load(f)
    temb = task_embeddings(cache["raw"], cache["flp"], u.region.values, u.side.values)
    rows = []
    for seed in a.seeds:
        m, pair_f1, name_f1 = evaluate(u, cache["geom"], temb, seed, use_rule=not a.no_rule)
        rows.append(dict(split=f"seed {seed}", **{k: round(v, 3) for k, v in m.items()}))
        print(rows[-1], flush=True)
        print("  по парам:", {k: round(v, 3) for k, v in pair_f1.items()}, flush=True)
        print("  по названиям:", {k: round(v, 3) for k, v in name_f1.items()}, flush=True)
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

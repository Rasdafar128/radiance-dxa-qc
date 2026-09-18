"""Nested, repeated, study-grouped cross-validation of the full ensemble.

Level-1: logistic regressions on (a) each foundation-model embedding, (b) geometry.
Level-2: per-task LR over level-1 OOF probabilities (+ other-hip context for hips).
Image quality_prob: LR over task probabilities (or noisy-OR), evaluated per image.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS = {"spine": ["ukl", "axis", "art"], "hip": ["rot", "roi"]}


def lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def fit_predict(Xtr, ytr, Xte, C):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), ytr.mean())
    return lr(C).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def inner_oof(X, y, groups, C, seed, k=5):
    p = np.zeros(len(y))
    splits = StratifiedGroupKFold(min(k, int(y.sum())) if y.sum() >= 2 else 2, shuffle=True,
                                  random_state=seed).split(X, y, groups)
    for tr, te in splits:
        p[te] = fit_predict(X[tr], y[tr], X[te], C)
    return p


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def other_hip(p, folders, sides):
    """Probability of the same task on the contralateral hip of the same study (nan if absent)."""
    s = pd.DataFrame({"p": p, "f": folders, "s": sides})
    lookup = {(f, sd): v for f, sd, v in zip(s.f, s.s, s.p)}
    return np.array([lookup.get((f, "L" if sd == "R" else "R"), np.nan) for f, sd in zip(s.f, s.s)])


def run(region, blocks, cfg, d, reps, seed0=0):
    m = (d.region == region) & d.labeled
    sub = d[m].reset_index(drop=True)
    idx = np.where(m)[0]
    groups = sub.folder.values
    tasks = TASKS[region]
    Y = {t: sub[f"y_{t}"].values.astype(int) for t in tasks}
    y_any = sub.y_any.values.astype(int)
    B = {name: X[idx] for name, X in blocks.items()}
    out = {f"p_{t}": np.zeros((reps, len(sub))) for t in tasks}
    out["p_any"] = np.zeros((reps, len(sub)))
    out["p_noisyor"] = np.zeros((reps, len(sub)))
    for r in range(reps):
        # stratify outer folds on "any" to keep every fold informative
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + r).split(sub, y_any, groups):
            task_te, task_tr = {}, {}
            for t in tasks:
                y = Y[t]
                l1_tr, l1_te = [], []
                for name, X in B.items():
                    C = cfg["C"].get(name, 0.01)
                    l1_tr.append(inner_oof(X[tr], y[tr], groups[tr], C, seed0 + r))
                    l1_te.append(fit_predict(X[tr], y[tr], X[te], C))
                Ltr, Lte = logit(np.stack(l1_tr, 1)), logit(np.stack(l1_te, 1))
                if cfg["stack"] == "mean":
                    ptr, pte = 1 / (1 + np.exp(-Ltr.mean(1))), 1 / (1 + np.exp(-Lte.mean(1)))
                else:
                    ptr = inner_oof(Ltr, y[tr], groups[tr], 1.0, seed0 + r)
                    pte = fit_predict(Ltr, y[tr], Lte, 1.0)
                if region == "hip" and cfg.get("context"):
                    # contralateral hip probability as an extra stacked feature
                    f_all, s_all = sub.folder.values, sub.side.values
                    o_tr = other_hip(ptr, f_all[tr], s_all[tr])
                    o_te = other_hip(pte, f_all[te], s_all[te])
                    Ztr = np.c_[logit(ptr), np.nan_to_num(logit(o_tr), nan=0.0), np.isnan(o_tr)]
                    Zte = np.c_[logit(pte), np.nan_to_num(logit(o_te), nan=0.0), np.isnan(o_te)]
                    ptr2 = inner_oof(Ztr, y[tr], groups[tr], 1.0, seed0 + r)
                    pte = fit_predict(Ztr, y[tr], Zte, 1.0)
                    ptr = ptr2
                task_tr[t], task_te[t] = ptr, pte
                out[f"p_{t}"][r, te] = pte
            Ttr = logit(np.stack([task_tr[t] for t in tasks], 1))
            Tte = logit(np.stack([task_te[t] for t in tasks], 1))
            out["p_any"][r, te] = fit_predict(Ttr, y_any[tr], Tte, 1.0)
            out["p_noisyor"][r, te] = 1 - np.prod(1 - np.stack([task_te[t] for t in tasks], 1), 1)
    return sub, out


def summarize(sub, out, region):
    rows = []
    for key, P in out.items():
        t = key[2:]
        y = sub.y_any.values if t in ("any", "noisyor") else sub[f"y_{t}"].values
        aucs = [roc_auc_score(y, p) for p in P]
        aps = [average_precision_score(y, p) for p in P]
        f1s = [max(f1_score(y, p >= th) for th in np.unique(p)) for p in P]
        rows.append(dict(region=region, target=t, npos=int(y.sum()), n=len(y), auc=np.mean(aucs),
                         auc_sd=np.std(aucs), ap=np.mean(aps), f1_best=np.mean(f1s)))
    return pd.DataFrame(rows)


def load_blocks(d, names, geom=True):
    blocks = {}
    for n in names:
        e = np.load(f"artifacts/emb_{n}.npy")
        ef = np.load(f"artifacts/emb_{n}_flip.npy")
        spine = (d.region == "spine").values[:, None]
        blocks[n] = np.where(spine, (e + ef) / 2, e)
    if geom:
        G = pd.read_pickle("artifacts/geom.pkl")
        blocks["geom_spine"] = G[[c for c in G.columns if c in SPINE_GEOM]].values.astype(float)
        blocks["geom_hip"] = G[[c for c in G.columns if c in HIP_GEOM]].values.astype(float)
    return blocks


SPINE_GEOM = ["angle_deg", "angle_end_deg", "dev_max_px", "dev_std_px", "curvature", "x0_rel", "pitch_px",
              "n_vertebrae", "crest_h_left", "crest_h_right", "crest_h_min", "crest_levels",
              "above_crest_vertebrae", "bottom_lat_frac", "top_lat_frac", "tophat_p999", "tophat_top_p99",
              "tophat_frac30", "sat_frac", "sat_max_blob", "n_sat_blobs", "p999", "spine_p99", "img_mean", "h_px"]
HIP_GEOM = ["shaft_w_px", "shaft_c_rel", "shaft_tilt_deg", "top_margin_sw", "lat_margin_sw", "bottom_len_sw",
            "below_lt_sw", "top_margin_mm", "lat_margin_mm", "below_lt_mm", "lt_prom_sw", "lt_rel_y",
            "medial_resid_max_sw", "fem_area_sw2", "fem_frac", "bone_frac", "fem_mean_int", "img_mean", "h_px", "ok"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=["dinov2_b", "rad_dino"])
    ap.add_argument("--no-geom", action="store_true")
    ap.add_argument("--stack", default="lr", choices=["lr", "mean"])
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = pd.read_pickle("artifacts/dataset.pkl")
    allb = load_blocks(d, a.emb, geom=not a.no_geom)
    res = []
    for region in ["spine", "hip"]:
        blocks = {k: v for k, v in allb.items() if not (k.startswith("geom_") and not k.endswith(region))}
        cfg = dict(C={"dinov2_s": 0.01, "dinov2_b": 0.03, "rad_dino": 0.03, "geom_spine": 0.1, "geom_hip": 0.1},
                   stack=a.stack, context=a.context)
        sub, out = run(region, blocks, cfg, d, a.reps)
        res.append(summarize(sub, out, region))
        np.save(f"artifacts/oof_{region}{a.tag}.npy", {k: v for k, v in out.items()}, allow_pickle=True)
    R = pd.concat(res)
    print(json.dumps(vars(a)))
    print(R.round(3).to_string(index=False))

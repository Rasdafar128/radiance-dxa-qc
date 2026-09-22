"""Spec-driven nested CV: each task = mean of logits of small per-block logistic regressions."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.cv_eval import fit_predict, logit, other_hip  # noqa: E402

SPECS = {
    "A_domain": {
        "ukl": [("geom", ["crest_levels", "above_crest_vertebrae", "bottom_lat_frac", "crest_h_min"], 1.0), ("emb", "dinov2_b", 0.03)],
        "axis": [("geom", ["angle_deg", "angle_end_deg"], 1.0)],
        "art": [("geom", ["tophat_top_p99", "tophat_p999", "sat_max_blob"], 1.0), ("emb", "rad_dino", 0.03)],
        "rot": [("emb", "rad_dino", 0.03), ("emb", "dinov2_b", 0.03)],
        "roi": [("geom", ["h_px", "lat_margin_mm", "top_margin_mm", "below_lt_mm", "ok"], 1.0)],
    },
    "B_domain_plus_emb": {
        "ukl": [("geom", ["crest_levels", "above_crest_vertebrae", "bottom_lat_frac", "crest_h_min"], 1.0), ("emb", "dinov2_b", 0.03)],
        "axis": [("geom", ["angle_deg", "angle_end_deg"], 1.0), ("emb", "dinov2_b", 0.03)],
        "art": [("geom", ["tophat_top_p99", "tophat_p999", "sat_max_blob"], 1.0), ("emb", "rad_dino", 0.03)],
        "rot": [("emb", "rad_dino", 0.03), ("emb", "dinov2_b", 0.03), ("geom", ["lt_prom_sw", "shaft_tilt_deg", "medial_resid_max_sw"], 1.0)],
        "roi": [("geom", ["h_px", "lat_margin_mm", "top_margin_mm", "below_lt_mm", "ok"], 1.0), ("emb", "rad_dino", 0.03)],
    },
    "C_geom_only": {
        "ukl": [("geom", ["crest_levels", "above_crest_vertebrae", "bottom_lat_frac", "crest_h_min"], 1.0)],
        "axis": [("geom", ["angle_deg"], 1.0)],
        "art": [("geom", ["tophat_top_p99"], 1.0)],
        "rot": [("geom", ["lt_prom_sw", "shaft_tilt_deg", "medial_resid_max_sw"], 1.0)],
        "roi": [("geom", ["h_px"], 1.0)],
    },
}
TASKS = {"spine": ["ukl", "axis", "art"], "hip": ["rot", "roi"]}


def block_matrix(d, G, E, spec_item):
    kind, what, C = spec_item
    if kind == "geom":
        return G[what].values.astype(float), C
    return E[what], C


def run(region, spec, d, G, E, reps, beta):
    m = ((d.region == region) & d.labeled).values
    sub = d[m].reset_index(drop=True)
    groups, y_any = sub.folder.values, sub.y_any.values.astype(int)
    tasks = TASKS[region]
    P = {t: np.zeros((reps, len(sub))) for t in tasks}
    P["noisyor"] = np.zeros((reps, len(sub)))
    for r in range(reps):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(sub, y_any, groups):
            probs = {}
            for t in tasks:
                y = sub[f"y_{t}"].values.astype(int)
                logits = []
                for item in spec[t]:
                    X, C = block_matrix(d, G, E, item)
                    X = X[m]
                    logits.append(logit(fit_predict(X[tr], y[tr], X[te], C)))
                p = 1 / (1 + np.exp(-np.mean(logits, 0)))
                probs[t] = p
            if region == "hip" and beta:
                for t in tasks:
                    full = np.full(len(sub), np.nan)
                    full[te] = probs[t]
                    o = other_hip(full, sub.folder.values, sub.side.values)[te]
                    probs[t] = 1 / (1 + np.exp(-(logit(probs[t]) + beta * np.nan_to_num(logit(o), nan=0.0))))
            for t in tasks:
                P[t][r, te] = probs[t]
            P["noisyor"][r, te] = 1 - np.prod([1 - probs[t] for t in tasks], 0)
    rows = []
    for k, M in P.items():
        y = y_any if k == "noisyor" else sub[f"y_{k}"].values
        rows.append(dict(region=region, target=k, npos=int(y.sum()),
                         auc=np.mean([roc_auc_score(y, p) for p in M]),
                         auc_sd=np.std([roc_auc_score(y, p) for p in M]),
                         ap=np.mean([average_precision_score(y, p) for p in M]),
                         f1_best=np.mean([max(f1_score(y, p >= th) for th in np.unique(p)) for p in M])))
    return pd.DataFrame(rows), sub, P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="A_domain")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.0)
    a = ap.parse_args()
    d = pd.read_pickle("artifacts/dataset.pkl")
    G = pd.read_pickle("artifacts/geom.pkl")
    spine = (d.region == "spine").values[:, None]
    E = {n: np.where(spine, (np.load(f"artifacts/emb_{n}.npy") + np.load(f"artifacts/emb_{n}_flip.npy")) / 2,
                     np.load(f"artifacts/emb_{n}.npy")) for n in ["dinov2_s", "dinov2_b", "rad_dino"]}
    res = []
    allP = {}
    for region in ["spine", "hip"]:
        R, sub, P = run(region, SPECS[a.spec], d, G, E, a.reps, a.beta)
        res.append(R)
        allP[region] = (sub, P)
    # overall image-level (both regions pooled), as the organizers will likely score it
    y = np.r_[allP["spine"][0].y_any.values, allP["hip"][0].y_any.values]
    Pp = np.c_[allP["spine"][1]["noisyor"], allP["hip"][1]["noisyor"]]
    pooled = dict(region="ALL", target="noisyor", npos=int(y.sum()),
                  auc=np.mean([roc_auc_score(y, p) for p in Pp]), auc_sd=np.std([roc_auc_score(y, p) for p in Pp]),
                  ap=np.mean([average_precision_score(y, p) for p in Pp]),
                  f1_best=np.mean([max(f1_score(y, p >= th) for th in np.unique(p)) for p in Pp]))
    R = pd.concat(res + [pd.DataFrame([pooled])])
    print(json.dumps(vars(a)))
    print(R.round(3).to_string(index=False))
    pd.to_pickle(allP, f"artifacts/oof_{a.spec}_b{a.beta}.pkl")

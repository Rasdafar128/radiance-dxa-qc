"""Compare model specs with the same repeated study-grouped CV on cached features.

python experiments/cv_variants.py --reps 5   (needs artifacts/features.pkl from scripts/train.py)
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.features import task_embeddings  # noqa: E402
from dxaqc.model import cross_val_oof  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402

G = config
VARIANTS = {
    "current": (config.SPEC, config.CONTEXT_BETA),
    "dinov2_only": ({
        "ukl": [("geom", G.SPINE_GEOM_UKL, 1.0), ("emb", "dinov2_b", 0.03)],
        "axis": [("geom", G.SPINE_GEOM_AXIS, 1.0), ("emb", "dinov2_b", 0.03)],
        "art": [("geom", G.SPINE_GEOM_ART, 1.0), ("emb", "dinov2_b", 0.03)],
        "rot": [("emb", "dinov2_b", 0.03), ("geom", G.HIP_GEOM_ROT, 1.0)],
        "roi": [("geom", G.HIP_GEOM_ROI, 1.0), ("emb", "dinov2_b", 0.03)],
    }, config.CONTEXT_BETA),
    "no_context": (config.SPEC, {"rot": 0.0, "roi": 0.0}),
    "context_1.0": (config.SPEC, {"rot": 1.0, "roi": 0.0}),
    "rot_C0.01": ({**config.SPEC, "rot": [("emb", "rad_dino", 0.01), ("emb", "dinov2_b", 0.01), ("geom", G.HIP_GEOM_ROT, 1.0)]}, config.CONTEXT_BETA),
    "rot_C0.1": ({**config.SPEC, "rot": [("emb", "rad_dino", 0.1), ("emb", "dinov2_b", 0.1), ("geom", G.HIP_GEOM_ROT, 1.0)]}, config.CONTEXT_BETA),
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--only", nargs="*")
a = ap.parse_args()
u = build_dataset(Path(a.data))
with open("artifacts/features.pkl", "rb") as f:
    c = pickle.load(f)
temb = task_embeddings(c["raw"], c["flp"], u.region.values, u.side.values)
geom = c["geom"]
rows = []
for name, (spec, beta) in VARIANTS.items():
    if a.only and name not in a.only:
        continue
    import dxaqc.model as M
    M.config.SPEC = spec
    M.config.CONTEXT_BETA = beta
    pooled_y, pooled_p = [], []
    res = {"variant": name}
    for reg in ("spine", "hip"):
        idx = np.where(((u.region == reg) & u.labeled).values)[0]
        sub = u.iloc[idx].reset_index(drop=True)
        Y = sub[[f"y_{t}" for t in config.TASKS[reg]] + ["y_any"]].astype(int)
        oofs = cross_val_oof(reg, geom.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in temb.items()},
                             Y, sub.folder.values, sub.side.values, reps=a.reps, seed=config.SEED)
        for t in config.TASKS[reg] + ["any"]:
            res[f"{reg}_{t}"] = np.mean([roc_auc_score(Y[f"y_{t}"], o[t].astype(float)) for o in oofs])
        pooled_y.append(Y["y_any"].values)
        pooled_p.append(np.stack([o["any"].astype(float).values for o in oofs]))
    y = np.concatenate(pooled_y)
    P = np.concatenate(pooled_p, 1)
    res["ALL_any"] = np.mean([roc_auc_score(y, p) for p in P])  # note: raw noisy-OR, not per-region calibrated
    rows.append(res)
    print(pd.DataFrame([res]).round(3).to_string(index=False), flush=True)
print(pd.DataFrame(rows).round(3).to_string(index=False))

"""Compare per-criterion feature recipes with repeated study-grouped CV.

python experiments/spec_search.py --reps 5 [--tasks rot roi ukl axis art]
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
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.features import task_embeddings  # noqa: E402
from dxaqc.model import cross_val_oof  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402

GEOM = {"ukl": config.SPINE_GEOM_UKL, "axis": config.SPINE_GEOM_AXIS, "art": config.SPINE_GEOM_ART,
        "rot": config.HIP_GEOM_ROT, "roi": config.HIP_GEOM_ROI}
REGION_OF = {"ukl": "spine", "axis": "spine", "art": "spine", "rot": "hip", "roi": "hip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--features", default="artifacts/features_all.pkl")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tasks", nargs="*", default=["ukl", "axis", "art", "rot", "roi"])
    ap.add_argument("--max-emb", type=int, default=2, help="max number of encoders per recipe")
    a = ap.parse_args()

    u = build_dataset(Path(a.data))
    with open(a.features, "rb") as f:
        cache = pickle.load(f)
    geom, raw, flp = cache["geom"], cache["raw"], cache["flp"]
    names = sorted(raw)
    temb = task_embeddings(raw, flp, u.region.values, u.side.values)
    print("encoders:", names, flush=True)

    rows = []
    for task in a.tasks:
        reg = REGION_OF[task]
        m = ((u.region == reg) & u.labeled).values
        idx = np.where(m)[0]
        sub = u.iloc[idx].reset_index(drop=True)
        g = geom.iloc[idx].reset_index(drop=True)
        e = {k: v[idx] for k, v in temb.items()}
        Y = sub[[f"y_{t}" for t in config.TASKS[reg]] + ["y_any"]].astype(int)
        y = Y[f"y_{task}"].to_numpy()
        combos = [()]
        for k in range(1, a.max_emb + 1):
            combos += list(itertools.combinations(names, k))
        for use_geom in (True, False):
            for combo in combos:
                if not combo and not use_geom:
                    continue
                spec = {t: config.SPEC[t] for t in config.TASKS[reg]}
                blocks = ([("geom", GEOM[task], 1.0)] if use_geom else []) + [("emb", n, 0.03) for n in combo]
                spec[task] = blocks
                oofs = cross_val_oof(reg, g, e, Y, sub.folder.values, sub.side.values,
                                     reps=a.reps, seed=config.SEED, spec=spec)
                auc = float(np.mean([roc_auc_score(y, o[task].astype(float)) for o in oofs]))
                ap_ = float(np.mean([average_precision_score(y, o[task].astype(float)) for o in oofs]))
                rows.append(dict(task=task, geom=use_geom, enc="+".join(combo) or "—", auc=round(auc, 3), ap=round(ap_, 3)))
                print(rows[-1], flush=True)
        best = pd.DataFrame([r for r in rows if r["task"] == task]).sort_values("auc", ascending=False)
        print(f"\n== best recipes for {task}\n{best.head(6).to_string(index=False)}\n", flush=True)
    pd.DataFrame(rows).to_csv("reports/spec_search.csv", index=False)


if __name__ == "__main__":
    main()

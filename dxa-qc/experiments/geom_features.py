import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.anatomy import canonical_hip
from dxaqc.geometry import hip_geometry, spine_geometry
d = pd.read_pickle("artifacts/dataset.pkl")
rows = []
for r in d.itertuples():
    g = spine_geometry(r.pixels) if r.region == "spine" else hip_geometry(canonical_hip(r.pixels, r.side))
    rows.append(g)
G = pd.DataFrame(rows, index=d.index)
G.to_pickle("artifacts/geom.pkl")
from sklearn.metrics import roc_auc_score
for region, tasks in [("spine", ["ukl", "axis", "art", "any"]), ("hip", ["rot", "roi", "any"])]:
    m = (d.region == region) & d.labeled
    X = G[m].dropna(axis=1, how="all")
    print("=== ", region)
    tab = {}
    for c in X.columns:
        x = X[c].fillna(X[c].median())
        if x.nunique() < 2: continue
        tab[c] = {t: max(roc_auc_score(d.loc[m, f"y_{t}"], x), 1 - roc_auc_score(d.loc[m, f"y_{t}"], x)) for t in tasks}
    print(pd.DataFrame(tab).T.round(3).sort_values(tasks[0], ascending=False).to_string())

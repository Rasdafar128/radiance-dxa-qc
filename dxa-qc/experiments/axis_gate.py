"""Does a minimal-angle gate for the axis violation improve F1 on OOF predictions?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.geometry import spine_geometry  # noqa: E402
from dxaqc.model import QualityModel  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402

u = build_dataset(Path("../data"))
O = pd.read_csv("reports/oof_predictions.csv", keep_default_na=False)
qm = QualityModel.load(Path("weights/qc_model.joblib"))
ang = {r.folder: spine_geometry(r.pixels).get("angle_deg", 0.0) for r in u[u.region == "spine"].itertuples()}
sp = O[O.region == "spine"].copy()
sp["angle"] = sp.folder.map(ang)
print(sp.groupby("y_axis").angle.describe().round(2))
for gate in [0, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
    pred_axis, pred_names_all = [], []
    for r in O.itertuples():
        probs = {t: float(getattr(r, f"p_{t}")) for t in config.TASKS[r.region]}
        probs["any"] = np.nan
        th = qm.thresholds[r.region]
        names = []
        if r.quality_class == 1:
            ratios = {t: probs[t] / th[t] for t in config.TASKS[r.region]}
            if r.region == "spine" and ang[r.folder] < gate:
                ratios["axis"] = 0.0
            viol = [t for t, v in ratios.items() if v >= 1.0] or [max(ratios, key=ratios.get)]
            names = sorted({config.VIOLATION_NAMES[(r.region, t)] for t in viol})
        pred_names_all.append(names)
    f1s = {}
    for nm in sorted(set(config.VIOLATION_NAMES.values())):
        yt = O.gt_violations.str.split(";").apply(lambda s: nm in s).astype(int)
        yp = pd.Series([nm in n for n in pred_names_all]).astype(int)
        f1s[nm] = f1_score(yt, yp, zero_division=0)
    print(f"gate {gate:>3}: axis F1 {f1s['Не выравнена ось позвоночника']:.3f}  macro-F1 {np.mean(list(f1s.values())):.3f}")

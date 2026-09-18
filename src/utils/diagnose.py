"""Описательный разбор сохранённых OOF; ничего не обучает и не выбирает.

python -m src.utils.diagnose artifacts/e2-dinov3-large artifacts/e2-dinov3-lbfgs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..solution.model import probability, targets
from .train import write_json


def diagnose(path):
    df, oof = pd.read_csv(path / "manifest.csv"), pd.read_csv(path / "oof.csv")
    with np.load(path / "features.npz", allow_pickle=False) as stored:
        assert np.array_equal(stored["image_ids"], df.image_id.to_numpy())
        x = stored["features"]
    truth = targets(df)
    models = [json.loads((path / f"fold_{fold}/model.json").read_text()) for fold in range(3)]
    adapted = models[0].get("recipe", "").startswith("dinov3-last2-bf16-")
    report = dict(run=str(path), targets={}, quality_gates=[
        {k: m["heads"][k]["threshold"] for k in ("spine_quality", "hip_quality")}
        if m.get("quality_gate", False) else None for m in models])
    for key in C.TARGETS:
        y, pred = oof[f"true_{key}"], oof[f"pred_{key}"] == 1
        folds = []
        for fold in range(3):
            head = models[fold]["heads"][key]
            train = (df.fold != fold) & np.isfinite(truth[key])
            # Adapted OOF features come from different encoders; never reuse them as train features.
            mean_train = float(probability(x[train], head).mean()) if "coef" in head and not adapted else None
            folds.append(dict(fold=fold, threshold=head["threshold"], C=head.get("C"), balanced=head.get("balanced"),
                              train_prevalence=float(np.mean(truth[key][train])),
                              mean_train_probability=mean_train))
        report["targets"][key] = dict(positives=int((y == 1).sum()),
                                       tp=int(((y == 1) & pred).sum()), fp=int(((y == 0) & pred).sum()),
                                       fn=int(((y == 1) & ~pred).sum()), folds=folds)
    false_alarms = oof[(oof.true_quality == 0) & (oof.pred_quality == 1)]
    report["quality_false_positives"] = len(false_alarms)
    report["sole_false_alarm_type"] = {t: int(((false_alarms[f"pred_{t}"] == 1) &
        (false_alarms[[f"pred_{k}" for k in C.TARGETS]].sum(axis=1) == 1)).sum()) for t in C.TARGETS}
    write_json(path / "diagnostics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.runs:
        print(json.dumps(diagnose(path), ensure_ascii=False, indent=2))

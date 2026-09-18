"""Повторная проверка Radiance: refit голов, пороги, происхождение и независимый расчёт метрик.

python -m src.utils.check_final --output research/results/final_audit.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .. import config as C
from ..solution.model import digest, fit_head, probability, targets
from .check_protocol import check
from .evaluate import evaluate
from .train import tune, write_json
from .train_blend import thresholds


def audit(path):
    recipe = json.loads((path / "recipe.json").read_text())
    df = pd.read_csv(path / "manifest.csv")
    assert recipe["weights"] == [.5, .5]
    parents = [C.ROOT / p for p in recipe["members"]]
    for parent, provenance in zip(parents, recipe["parents"]):
        check(parent)
        assert digest(parent / "recipe.json") == provenance["recipe_sha256"]
        assert digest(parent / "features.npz") == provenance["features_sha256"]
        assert (parent / "manifest.csv").read_bytes() == (path / "manifest.csv").read_bytes()
        parent_recipe = json.loads((parent / "recipe.json").read_text())
        assert parent_recipe["id"].startswith("frozen-")
        with np.load(parent / "features.npz", allow_pickle=False) as cache:
            x = cache["features"]
        for fold in range(3):
            train = df[df.fold != fold].reset_index(drop=True)
            features = x[df.fold != fold]
            split = pd.read_csv(parent / f"inner_{fold}.csv")
            selected, scores, _, _ = tune(train, features, split)
            stored = json.loads((parent / f"fold_{fold}/selection.json").read_text())
            historical = {key: dict(value, solver=value.get("solver", "liblinear"))
                          for key, value in stored["parameters"].items()}
            assert selected == historical
            saved = pd.read_csv(parent / f"fold_{fold}/inner_predictions.csv").set_index("image_id").reindex(train.image_id)
            np.testing.assert_allclose(scores, saved[scores.columns], rtol=1e-10, atol=1e-10, equal_nan=True)
            heads = json.loads((parent / f"fold_{fold}/model.json").read_text())["heads"]
            for key, y in targets(train).items():
                p = selected[key]
                known = np.isfinite(y)
                refit = fit_head(features[known], y[known], p["C"], p["balanced"], p["solver"])
                np.testing.assert_allclose(probability(x, refit), probability(x, heads[key]), rtol=1e-10, atol=1e-10)
    check(path)
    for fold in [0, 1, 2, "final"]:
        train = df if fold == "final" else df[df.fold != fold]
        folder = "final" if fold == "final" else f"fold_{fold}"
        filename = "final_inner_predictions.csv" if fold == "final" else f"{folder}/inner_predictions.csv"
        scores = [pd.read_csv(p / filename).set_index("image_id").reindex(train.image_id) for p in parents]
        average = (scores[0] + scores[1]) / 2
        saved = pd.read_csv(path / folder / "inner_predictions.csv").set_index("image_id").reindex(train.image_id)
        np.testing.assert_allclose(average, saved[average.columns], rtol=1e-12, atol=1e-12, equal_nan=True)
        metadata = json.loads((path / folder / "model.json").read_text())
        assert thresholds(train, saved) == metadata["heads"]
        for i, parent in enumerate(parents):
            assert (path / folder / f"member_{i}/model.json").read_bytes() == (parent / folder / "model.json").read_bytes()
    oof = pd.read_csv(path / "oof.csv")
    stored = json.loads((path / "metrics.json").read_text())
    assert evaluate(oof, stored["resamples"]) == stored
    quality = dict(f1=float(f1_score(oof.true_quality, oof.pred_quality)),
                   roc_auc=float(roc_auc_score(oof.true_quality, oof.prob_quality)))
    for key, value in quality.items():
        assert np.isclose(value, stored["metrics"]["quality_all"][key], atol=1e-14)
    per_type = {}
    for key, region in zip(C.TARGETS, C.TARGET_REGIONS):
        subset = oof[oof.anatomical_region == region]
        per_type[key] = float(f1_score(subset[f"true_{key}"], subset[f"pred_{key}"], zero_division=0))
    macro = float(np.mean(list(per_type.values())))
    assert np.isclose(macro, stored["macro_f1"], atol=1e-14)
    tn, fp, fn, tp = confusion_matrix(oof.true_quality, oof.pred_quality, labels=[0, 1]).ravel()
    return dict(run=path.name, images=len(df), studies=int(df.study.nunique()),
                recipe_sha256=digest(path / "recipe.json"), oof_sha256=digest(path / "oof.csv"),
                quality=quality, macro_f1=macro, per_type_f1=per_type,
                confusion={k: int(v) for k, v in zip(("tn", "fp", "fn", "tp"), (tn, fp, fn, tp))},
                checks=["unique pixels and complete study-disjoint outer OOF",
                        "parent recipe, feature, source and manifest hashes",
                        "inner tuning and outer-training head refit reproduced for both parents",
                        "blend members, 0.5/0.5 inner probabilities and inner-only thresholds",
                        "saved outer predictions reproduced from heads",
                        "all published metrics and 2000 study-bootstrap intervals reproduced",
                        "F1, AUC, macro-F1 and confusion independently checked with sklearn"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "delivery-checks/final_audit.json")
    args = parser.parse_args()
    result = dict(runs=[audit(C.ARTIFACTS / name) for name in ("e5-blend", "e5-blend-split137")],
                  independent_test=False, patient_independence_confirmed=False,
                  limitation="Development model selection uses OOF; bootstrap excludes model-selection uncertainty.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))

"""Проверить границы исследований, происхождение и полноту сохранённого OOF.

python -m src.utils.check_protocol artifacts/e2-dinov3-lbfgs
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ..solution.model import HEADS, Blend, Model, digest, probability, targets
from .train import predictions, write_json
from .train_blend import thresholds


def check(path):
    assert not (path / "invalid.json").exists(), "Invalidated experiment"
    df = pd.read_csv(path / "manifest.csv")
    recipe = json.loads((path / "recipe.json").read_text())
    assert digest(path / "manifest.csv") == recipe["manifest_sha256"]
    assert not df.image_id.duplicated().any() and not df.pixel_sha256.duplicated().any()
    assert df.groupby("study").fold.nunique().eq(1).all()
    oof = pd.read_csv(path / "oof.csv")
    with np.load(path / "features.npz", allow_pickle=False) as cache:
        assert np.array_equal(cache["image_ids"], df.image_id.to_numpy())
        features = cache["features"]
    assert not oof.image_id.duplicated().any()
    pd.testing.assert_frame_equal(df[["image_id", "study", "fold"]].sort_values("image_id").reset_index(drop=True),
                                  oof[["image_id", "study", "fold"]].sort_values("image_id").reset_index(drop=True))
    for fold in range(3):
        train, valid = df[df.fold != fold], df[df.fold == fold]
        inner = pd.read_csv(path / f"inner_{fold}.csv")
        assert set(inner.study) == set(train.study)
        assert not inner.study.duplicated().any() and set(inner.fold) == {0, 1}
        assert not set(inner.study) & set(valid.study)
        destination = path / f"fold_{fold}"
        metadata = json.loads((destination / "model.json").read_text())
        assert metadata["training_partition"] == f"outer_train_{fold}"
        assert set(pd.read_csv(destination / "inner_predictions.csv").image_id) == set(train.image_id)
        saved = pd.read_csv(destination / "oof.csv")
        assert set(saved.image_id) == set(valid.image_id)
        if metadata.get("backbone") == "blend":
            from .train_blend import cached_model

            model = Blend([cached_model(destination / f"member_{i}") for i in range(2)],
                          metadata["heads"], metadata["quality_gate"])
        else:
            model = Model.__new__(Model)
        model.metadata, model.heads = metadata, metadata["heads"]
        reproduced = predictions(model, valid, features[df.fold == fold])
        pd.testing.assert_frame_equal(saved, reproduced, check_exact=False, rtol=1e-12, atol=1e-12)
        completed = json.loads((destination / "complete.json").read_text())
        assert completed["oof_sha256"] == digest(destination / "oof.csv")
        selection = json.loads((destination / "selection.json").read_text())["parameters"]
        assert all(metadata["heads"][k]["threshold"] == p["threshold"] for k, p in selection.items())
        repeats = recipe.get("inner_repeats", 1)
        for repeat in range(1, repeats):
            repeated_split = pd.read_csv(path / f"inner_{fold}_repeat{repeat}.csv")
            assert set(repeated_split.study) == set(train.study)
            assert not repeated_split.study.duplicated().any() and set(repeated_split.fold) == {0, 1}
            assert not set(repeated_split.study) & set(valid.study)
        if repeats > 1:
            repeated = pd.read_csv(destination / "repeated_inner_predictions.csv")
            assert not repeated.duplicated(["repeat", "image_id"]).any()
            assert set(repeated.repeat) == set(range(repeats))
            assert repeated.groupby("image_id").size().eq(repeats).all()
            assert set(repeated.image_id) == set(train.image_id)
            index = pd.MultiIndex.from_product([range(repeats), train.image_id], names=["repeat", "image_id"])
            scores = repeated.set_index(["repeat", "image_id"]).reindex(index)
            expected_thresholds = thresholds(pd.concat([train] * repeats, ignore_index=True), scores)
            assert all(metadata["heads"][key]["threshold"] == value["threshold"]
                       for key, value in expected_thresholds.items())
        if recipe["id"].startswith("dinov3-last2-bf16-"):
            check_adaptation(path, fold, train, valid, inner, destination)
    for relative, checksum in recipe["code_sha256"].items():
        assert digest(path / "source" / relative) == checksum
    if recipe["id"].startswith("dinov3-last2-bf16-"):
        check_adaptation(path, "final", df, df.iloc[:0], pd.read_csv(path / "inner_final.csv"), path / "final")
    result = dict(images=len(df), studies=int(df.study.nunique()), outer_folds=3, inner_folds=2,
                  inner_repeats=recipe.get("inner_repeats", 1), adapted_partitions_checked=recipe["id"].startswith("dinov3-last2-bf16-"),
                  checks="manifest and source hashes, unique pixels, study disjointness, inner-only selection rows, complete OOF, OOF reproduced from saved heads",
                  limitation="Study grouping only: anonymized data cannot establish patient independence; model selection uses development OOF")
    write_json(path / "protocol_checks.json", result)
    print(json.dumps(dict(run=str(path), **result), indent=2))


def check_adaptation(path, fold, train, valid, split, destination):
    """Every checkpoint's training rows must match its nested partition exactly."""
    recipe = json.loads((path / "recipe.json").read_text())
    details = json.loads((destination / "selection.json").read_text())
    choices = []
    for lr in (1e-5, 3e-5):
        losses, epochs = [], []
        for inner_fold in (0, 1):
            job = path / "jobs" / f"{fold}_lr{lr}_inner{inner_fold}"
            part = json.loads((job / "partition.json").read_text())
            assert part["code_sha256"] == recipe["code_sha256"]["src/utils/finetune.py"]
            assert part["source_encoder_sha256"] == recipe["source_encoder_sha256"]
            groups = split.set_index("study").fold
            expected_train = train[train.study.map(groups) != inner_fold]
            expected_valid = train[train.study.map(groups) == inner_fold]
            assert set(part["train_ids"]) == set(expected_train.image_id)
            assert set(part["valid_ids"]) == set(expected_valid.image_id)
            assert not (set(part["train_studies"]) | set(part["valid_studies"])) & set(valid.study)
            labels = targets(expected_train)
            np.testing.assert_allclose(part["positive_weight"],
                [np.sum(labels[key] == 0) / np.sum(labels[key] == 1) for key in HEADS], rtol=1e-6)
            assert part["known_counts"] == [int(np.isfinite(labels[key]).sum()) for key in HEADS]
            saved = pd.read_csv(job / "predictions.csv")
            assert set(saved.image_id) == set(expected_valid.image_id)
            assert saved.image_id.tolist() == part["valid_ids"]
            heads = json.loads((job / "heads.json").read_text())
            features = np.load(job / "validation_features.npy")
            for key in HEADS:
                np.testing.assert_allclose(saved[key], probability(features, heads[key]), rtol=1e-12, atol=1e-12)
            history = json.loads((job / "history.json").read_text())
            completed = json.loads((job / "complete.json").read_text())
            assert completed["partition_sha256"] == digest(job / "partition.json")
            best = min(history, key=lambda row: (row["validation_loss"], row["epoch"]))
            assert completed["best_epoch"] == best["epoch"]
            assert completed["best_loss"] == best["validation_loss"]
            assert digest(job / "best.pt") == completed["best_sha256"]
            losses.append(best["validation_loss"])
            epochs.append(best["epoch"])
        choices.append((float(np.mean(losses)), lr, int(np.floor(np.median(epochs)+.5))))
    _, lr, epochs = min(choices)
    assert details["chosen_lr"] == lr and details["epochs"] == epochs
    refit = json.loads((path / "jobs" / f"{fold}_refit" / "partition.json").read_text())
    assert set(refit["train_ids"]) == set(train.image_id) and refit["valid_ids"] == []
    assert refit["refit_epochs"] == epochs and refit["lr"] == lr
    scores = pd.read_csv(destination / "inner_predictions.csv").set_index("image_id").reindex(train.image_id)
    selected_scores = pd.concat([pd.read_csv(path / "jobs" / f"{fold}_lr{lr}_inner{k}" / "predictions.csv")
                                 for k in (0, 1)]).set_index("image_id").reindex(train.image_id)
    np.testing.assert_allclose(scores[HEADS], selected_scores[HEADS], rtol=1e-12, atol=1e-12)
    assert thresholds(train, scores) == details["parameters"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.runs:
        check(path)

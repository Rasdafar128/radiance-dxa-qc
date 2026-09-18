"""Проверить границы исследований, происхождение и полноту сохранённого OOF.

python -m src.utils.check_protocol artifacts/e2-dinov3-lbfgs
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ..solution.model import Blend, Model, digest
from .train import predictions, write_json


def check(path):
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
    for relative, checksum in recipe["code_sha256"].items():
        assert digest(path / "source" / relative) == checksum
    result = dict(images=len(df), studies=int(df.study.nunique()), outer_folds=3, inner_folds=2,
                  checks="manifest and source hashes, unique pixels, study disjointness, inner-only selection rows, complete OOF, OOF reproduced from saved heads",
                  limitation="Study grouping only: anonymized data cannot establish patient independence; model selection uses development OOF")
    write_json(path / "protocol_checks.json", result)
    print(json.dumps(dict(run=str(path), **result), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.runs:
        check(path)

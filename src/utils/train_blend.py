"""Radiance: среднее двух frozen-рецептов с порогами по сохранённым внутренним OOF.

python -m src.utils.train_blend artifacts/reproduced-dino artifacts/reproduced-mii --output artifacts/reproduced-radiance
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .. import config as C
from ..solution.model import HEADS, Blend, Model, digest, targets
from .evaluate import evaluate
from .train import predictions, write_json


def cached_model(path):
    payload = json.loads((path / "model.json").read_text())
    if payload["schema"] != 1:
        raise ValueError("Blend members must be individual frozen models")
    if digest(path / "encoder.pt") != payload["encoder_sha256"]:
        raise ValueError("Member encoder checksum mismatch")
    model = Model.__new__(Model)
    model.heads, model.metadata = payload.pop("heads"), payload
    return model


def thresholds(df, scores):
    heads = {"region": dict(threshold=.5)}
    for key, y in targets(df).items():
        if key not in C.TARGETS:
            continue
        known = np.isfinite(y)
        _, threshold = max((f1_score(y[known], scores[key].to_numpy()[known] >= t, zero_division=0), float(t))
                           for t in np.arange(1, 20) / 20)
        heads[key] = dict(threshold=threshold)
    for region, key in ((C.REGION_SPINE, "spine_quality"), (C.REGION_FEMUR, "hip_quality")):
        mask = (df.anatomical_region == region).to_numpy()
        types = [t for t, r in zip(C.TARGETS, C.TARGET_REGIONS) if r == region]
        any_type = np.logical_or.reduce([scores[t].to_numpy()[mask] >= heads[t]["threshold"] for t in types])
        _, threshold = max((f1_score(df.quality_class.to_numpy()[mask],
                            any_type & (scores[key].to_numpy()[mask] >= t), zero_division=0), float(t))
                           for t in np.arange(20) / 20)
        heads[key] = dict(threshold=threshold)
    assert set(heads) == set(HEADS)
    return heads


def run(first, second, output, resamples):
    parents = [first, second]
    if any((p / "invalid.json").exists() for p in parents):
        raise ValueError("Cannot blend an invalidated experiment")
    df = pd.read_csv(first / "manifest.csv")
    repeats = [json.loads((p / "recipe.json").read_text()).get("inner_repeats", 1) for p in parents]
    assert repeats[0] == repeats[1]
    assert (first / "manifest.csv").read_bytes() == (second / "manifest.csv").read_bytes()
    assert not df.image_id.duplicated().any() and not df.pixel_sha256.duplicated().any()
    assert df.groupby("study").fold.nunique().eq(1).all()
    for name in [0, 1, 2, "final"]:
        assert (first / f"inner_{name}.csv").read_bytes() == (second / f"inner_{name}.csv").read_bytes()
        train = df if name == "final" else df[df.fold != name]
        inner = pd.read_csv(first / f"inner_{name}.csv")
        assert set(inner.study) == set(train.study) and not inner.study.duplicated().any()
    code = [C.ROOT / "src" / p for p in ("config.py", "solution/model.py", "solution/dicom.py",
            "solution/medimage_davit.py", "utils/train_blend.py", "utils/train.py", "utils/evaluate.py")]
    recipe = dict(id="frozen-probability-mean-v1", members=[str(p) for p in parents], inner_repeats=repeats[0],
                  weights=[.5, .5], quality_gate=True, resamples=resamples,
                  parents=[dict(recipe_sha256=digest(p / "recipe.json"), features_sha256=digest(p / "features.npz")) for p in parents],
                  manifest_sha256=digest(first / "manifest.csv"),
                  code_sha256={str(p.relative_to(C.ROOT)): digest(p) for p in code},
                  selection="Thresholds on averaged inner OOF only; development architecture selection")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "recipe.json").exists() and json.loads((output / "recipe.json").read_text()) != recipe:
        raise ValueError("Run configuration changed; choose a new output")
    write_json(output / "recipe.json", recipe)
    for p in code:
        destination = output / "source" / p.relative_to(C.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    shutil.copyfile(first / "manifest.csv", output / "manifest.csv")
    features = []
    for p in parents:
        with np.load(p / "features.npz", allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["image_ids"], df.image_id.to_numpy())
            features.append(cache["features"])
    x = np.concatenate(features, axis=1)
    np.savez_compressed(output / "features.npz", features=x, image_ids=df.image_id.to_numpy(dtype=str))

    for name in [0, 1, 2, "final"]:
        train = df if name == "final" else df[df.fold != name]
        inner_file = "final_inner_predictions.csv" if name == "final" else f"fold_{name}/inner_predictions.csv"
        inner = [pd.read_csv(p / inner_file).set_index("image_id").reindex(train.image_id) for p in parents]
        assert all(set(v.index) == set(train.image_id) and not v.isna().all(axis=1).any() for v in inner)
        scores = (inner[0] + inner[1]) / 2
        repeated_scores = None
        if repeats[0] > 1:
            filename = "final_repeated_inner_predictions.csv" if name == "final" else f"fold_{name}/repeated_inner_predictions.csv"
            index = pd.MultiIndex.from_product([range(repeats[0]), train.image_id], names=["repeat", "image_id"])
            repeated = [pd.read_csv(p / filename).set_index(["repeat", "image_id"]).reindex(index) for p in parents]
            assert all(v.index.is_unique and len(v) == len(train) * repeats[0] for v in repeated)
            repeated_scores = (repeated[0] + repeated[1]) / 2
            heads = thresholds(pd.concat([train] * repeats[0], ignore_index=True), repeated_scores)
        else:
            heads = thresholds(train, scores)
        folder = "final" if name == "final" else f"fold_{name}"
        sources = [p / folder for p in parents]
        members = [cached_model(p) for p in sources]
        assert all(m.metadata["training_partition"] == ("all_labeled" if name == "final" else f"outer_train_{name}") for m in members)
        model = Blend(members, heads)
        model.metadata.update(training=recipe, training_partition="all_labeled" if name == "final" else f"outer_train_{name}")
        destination = output / folder
        destination.mkdir(exist_ok=True)
        for i, source in enumerate(sources):
            member = destination / f"member_{i}"
            member.mkdir(exist_ok=True)
            shutil.copyfile(source / "model.json", member / "model.json")
            if not (member / "encoder.pt").exists():
                os.link(source / "encoder.pt", member / "encoder.pt")
        write_json(destination / "model.json", dict(model.metadata, heads=heads))
        write_json(destination / "selection.json", dict(parameters=heads, selection="averaged_inner_oof"))
        scores.to_csv(destination / "inner_predictions.csv", index_label="image_id")
        if repeated_scores is not None:
            repeated_scores.to_csv(destination / "repeated_inner_predictions.csv")
        shutil.copyfile(first / f"inner_{name}.csv", output / f"inner_{name}.csv")
        for r in range(1, repeats[0]):
            filename = f"inner_{name}_repeat{r}.csv"
            assert (first / filename).read_bytes() == (second / filename).read_bytes()
            shutil.copyfile(first / filename, output / filename)
        if name != "final":
            valid = df.fold == name
            predictions(model, df[valid], x[valid]).to_csv(destination / "oof.csv", index=False)
            write_json(destination / "complete.json", dict(oof_sha256=digest(destination / "oof.csv")))
        print(f"Blend {folder} completed", flush=True)

    oof = pd.concat([pd.read_csv(output / f"fold_{k}/oof.csv") for k in range(3)], ignore_index=True)
    assert len(oof) == len(df) and oof.image_id.nunique() == len(df)
    oof.to_csv(output / "oof.csv", index=False)
    report = evaluate(oof, resamples)
    write_json(output / "metrics.json", report)
    write_json(output / "complete.json", dict(recipe_sha256=digest(output / "recipe.json"), metrics_sha256=digest(output / "metrics.json")))
    print(json.dumps(dict(quality_f1=report["metrics"]["quality_all"]["f1"],
                         quality_auc=report["metrics"]["quality_all"]["roc_auc"], macro_f1=report["macro_f1"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    run(args.first, args.second, args.output, args.resamples)

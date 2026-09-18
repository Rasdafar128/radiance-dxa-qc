"""E0 на фиксированных исследованиях — вложенная настройка, OOF и автономные веса.

    python -m src.utils.train --device cuda --output artifacts/e0

Повторный запуск принимает только совпадающий рецепт, данные и разбиения.
Завершённые фолды не пересчитываются; сохранённый кэш содержит только frozen-признаки.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .. import config as C
from ..solution.model import BACKBONES, Model, digest, fit_head, probability, targets
from . import data
from .evaluate import evaluate


def sha(path):
    return digest(path)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inner_folds(df):
    group_targets = pd.DataFrame(targets(df))[C.TARGETS].assign(study=df.study.to_numpy())
    groups = group_targets.groupby("study").max().fillna(0)
    rng, best = np.random.default_rng(42), None
    for _ in range(1000):
        assignment = np.empty(len(groups), dtype=int)
        assignment[rng.permutation(len(groups))] = np.arange(len(groups)) % 2
        totals = np.stack([groups.to_numpy()[assignment == k].sum(axis=0) for k in range(2)])
        score = float(((totals - groups.sum().to_numpy() / 2)**2 / np.maximum(groups.sum().to_numpy(), 1)).sum())
        score += 100 * int((totals == 0).sum())
        if best is None or score < best[0]:
            best = (score, assignment.copy())
    return pd.DataFrame({"study": groups.index, "fold": best[1]})


def tune(df, x, split):
    assignments = df.study.map(split.set_index("study").fold).to_numpy()
    chosen, predictions, balance = {}, {}, {}
    for key, y in targets(df).items():
        known = np.isfinite(y)
        balance[key] = [{"fold": k, "positive": int((y[assignments == k] == 1).sum()),
                         "negative": int((y[assignments == k] == 0).sum())} for k in range(2)]
        usable = all(np.unique(y[known & (assignments == k)]).size == 2 for k in range(2))
        if key == "region" or not usable:
            chosen[key] = dict(C=0.1, balanced=True, threshold=0.5, selection="fixed")
            if not usable:
                chosen[key]["limitation"] = "An inner partition lacks both classes"
            continue
        best = None
        for regularization in (0.01, 0.1, 1.0):
            for balanced in (False, True):
                p = np.full(len(df), np.nan)
                for k in range(2):
                    train, valid = known & (assignments != k), known & (assignments == k)
                    h = fit_head(x[train], y[train], regularization, balanced)
                    p[valid] = probability(x[valid], h)
                if key.endswith("quality"):
                    score, threshold = roc_auc_score(y[known], p[known]), 0.5
                else:
                    score, threshold = max((f1_score(y[known], p[known] >= t, zero_division=0), float(t))
                                           for t in np.arange(1, 20) / 20)
                # Строгое > оставляет меньший C; порог при равенстве берётся больший.
                if best is None or score > best[0]:
                    best = (score, dict(C=regularization, balanced=balanced, threshold=threshold,
                                        selection="inner_oof"), p.copy())
        chosen[key], predictions[key] = best[1], best[2]
    return chosen, pd.DataFrame(predictions), balance


def manifest(audit):
    summary = json.loads((audit / "summary.json").read_text())
    if summary["primary_candidates"]["label_policy"] != "criteria_v2":
        raise ValueError("Re-run audit for criteria_v2")
    candidates = pd.read_csv(audit / "training_candidates.csv")
    df = data.table()
    df = df[df.quality_class.notna()].copy()
    df = df.merge(candidates[["path_to_study", "image_id", "pixel_sha256", "file_sha256", *C.TARGETS]],
                  on="path_to_study", validate="one_to_one")
    fresh = targets(df)
    for target in C.TARGETS:
        if not np.array_equal(df[target].to_numpy(), fresh[target], equal_nan=True):
            raise ValueError("Labels changed: rerun audit before training")
    for row in df.itertuples():
        path = C.DATA / row.path_to_study
        a = pydicom.dcmread(path).pixel_array
        digest = hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()
        if sha(path) != row.file_sha256 or digest != row.pixel_sha256:
            raise ValueError(f"DICOM changed since audit: {row.image_id}")
    df = df.merge(pd.read_csv(audit / "candidate_folds_primary_3.csv"), on="study", validate="many_to_one")
    if len(df) != 249 or df.study.nunique() != 100 or set(df.fold) != {0, 1, 2}:
        raise ValueError("Unexpected criteria_v2 data or folds")
    if df.pixel_sha256.duplicated().any() or df.groupby("study").fold.nunique().max() != 1:
        raise ValueError("Duplicate pixels or group leakage")
    return df.sort_values("image_id").reset_index(drop=True)


def predictions(model, df, x):
    result, scores = model.classify(x)
    out = df[["image_id", "study", "path_to_study", "anatomical_region", "side", "fold"]].reset_index(drop=True).copy()
    out["predicted_region"] = result.anatomical_region
    out["true_quality"] = df.quality_class.to_numpy()
    out["pred_quality"] = result.quality_class
    out["prob_quality"] = result.quality_prob
    truth = targets(df)
    for target, region in zip(C.TARGETS, C.TARGET_REGIONS):
        out[f"true_{target}"] = truth[target]
        routed = (result.anatomical_region == region).to_numpy()
        out[f"prob_{target}"] = np.where(routed, scores[target], 0)
        out[f"pred_{target}"] = routed & (scores[target] >= model.heads[target]["threshold"])
    return out


def run(output, device, resamples, backbone="b0", weights=None):
    started = perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    df = manifest(C.ARTIFACTS / "audit")
    manifest_text = df.to_csv(index=False)
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "torchvision", "numpy", "pandas", "pydicom", "Pillow", "scikit-learn")}
    source = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
    encoder_config = None
    if backbone != "b0":
        if weights is None:
            raise ValueError("Use --weights with a downloaded local snapshot")
        source = json.loads((weights / "source.json").read_text())
        for name, checksum in source["sha256"].items():
            if sha(weights / name) != checksum:
                raise ValueError(f"Source checksum mismatch: {name}")
        encoder_config = json.loads((weights / "config.json").read_text())
        if backbone == "medsiglip":
            encoder_config = encoder_config["vision_config"]
        versions.update({p: importlib.metadata.version(p) for p in
                         ("transformers", "safetensors", "huggingface-hub")})
    code = [C.ROOT / "src" / p for p in ("config.py", "solution/model.py", "solution/dicom.py",
                                          "utils/train.py", "utils/evaluate.py", "utils/data.py")]
    recipe = dict(id="E0-b0-linear-v1" if backbone == "b0" else f"frozen-{backbone}-v1",
                  backbone=backbone, source_weights=source, label_policy="criteria_v2", seed=42, resamples=resamples,
                  manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest(),
                  code_sha256={str(p.relative_to(C.ROOT)): sha(p) for p in code}, versions=versions,
                  python=platform.python_version(), device=device,
                  hypothesis=f"Frozen {backbone}, nested L2 logistic heads, complete inference path",
                  selection="Development comparison; no independent test claim")
    recipe_path = output / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("Run configuration changed; choose a new output directory")
    write_json(recipe_path, recipe)
    (output / "manifest.csv").write_text(manifest_text)
    splits = {}
    # Все назначения фиксируются до первого обращения к энкодеру и головам.
    for name in [0, 1, 2, "final"]:
        train = df if name == "final" else df[df.fold != name]
        splits[name] = inner_folds(train.reset_index(drop=True))
        splits[name].to_csv(output / f"inner_{name}.csv", index=False)

    cache = output / "features.npz"
    if cache.exists():
        model = Model(device=device, backbone=backbone, encoder_config=encoder_config)
        model.encoder.load_state_dict(torch.load(output / "encoder.pt", map_location="cpu", weights_only=True))
        with np.load(cache, allow_pickle=False) as stored:
            if not np.array_equal(stored["image_ids"], df.image_id.to_numpy()):
                raise ValueError("Feature cache does not match manifest order")
            x = stored["features"]
    else:
        model = Model(device=device, pretrained=True, backbone=backbone, weights=weights)
        x = model.features([C.DATA / p for p in df.path_to_study])
        torch.save({k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}, output / "encoder.pt")
        np.savez_compressed(cache, features=x, image_ids=df.image_id.to_numpy(dtype=str))
    model.metadata["training"] = recipe
    if x.shape != (len(df), model.metadata["dimensions"]) or not np.isfinite(x).all():
        raise ValueError("Invalid frozen feature cache")
    model.metadata["source_weights"] = source
    print(f"Frozen features: {x.shape}, elapsed {perf_counter() - started:.1f}s", flush=True)

    for fold in range(3):
        destination = output / f"fold_{fold}"
        destination.mkdir(exist_ok=True)
        if (destination / "complete.json").exists():
            completed = json.loads((destination / "complete.json").read_text())
            if completed["oof_sha256"] != sha(destination / "oof.csv"):
                raise ValueError("Completed fold checksum mismatch")
            continue
        train, valid = (df.fold != fold).to_numpy(), (df.fold == fold).to_numpy()
        if set(df[train].study) & set(df[valid].study):
            raise ValueError("Study leakage")
        parameters, inner, balance = tune(df[train].reset_index(drop=True), x[train], splits[fold])
        write_json(destination / "selection.json", dict(parameters=parameters, balance=balance))
        inner.insert(0, "image_id", df[train].image_id.to_numpy())
        inner.to_csv(destination / "inner_predictions.csv", index=False)
        model.fit(df[train], features=x[train], parameters=parameters, verbose=False)
        model.metadata["training_partition"] = f"outer_train_{fold}"
        model.save(destination, encoder_path=output / "encoder.pt")
        reloaded = Model.load(destination, device)
        before, _ = model.classify(x[valid])
        after, _ = reloaded.classify(x[valid])
        pd.testing.assert_frame_equal(before, after)
        del reloaded
        predictions(model, df[valid], x[valid]).to_csv(destination / "oof.csv", index=False)
        write_json(destination / "complete.json", dict(oof_sha256=sha(destination / "oof.csv")))
        print(f"Outer fold {fold} completed", flush=True)

    oof = pd.concat([pd.read_csv(output / f"fold_{k}" / "oof.csv") for k in range(3)], ignore_index=True)
    assert len(oof) == len(df) and oof.image_id.nunique() == len(df)
    oof.to_csv(output / "oof.csv", index=False)
    report = evaluate(oof, resamples)
    # Чувствительность фиксированных OOF, без выбора стороны по метрике.
    hips = df[df.anatomical_region == C.REGION_FEMUR]
    disputed = [g for g, pair in hips.groupby("study") if len(pair) == 2 and pair.violation_type.nunique() > 1]
    subset = oof[~((oof.anatomical_region == C.REGION_FEMUR) & oof.study.isin(disputed))].reset_index(drop=True)
    report["side_sensitivity"] = dict(excluded_paired_studies=len(disputed), fixed_oof=evaluate(subset, 0))
    write_json(output / "metrics.json", report)
    parameters, inner, balance = tune(df, x, splits["final"])
    write_json(output / "final_selection.json", dict(parameters=parameters, balance=balance))
    inner.insert(0, "image_id", df.image_id.to_numpy())
    inner.to_csv(output / "final_inner_predictions.csv", index=False)
    model.fit(df, features=x, parameters=parameters, verbose=False)
    model.metadata["training_partition"] = "all_labeled"
    model.save(output / "final", encoder_path=output / "encoder.pt")
    hardware = dict(python=platform.python_version(), platform=platform.platform(), versions=versions,
                    elapsed_seconds=perf_counter() - started,
                    peak_vram_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0)
    if device.startswith("cuda"):
        hardware["gpu"] = torch.cuda.get_device_name()
        hardware["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    write_json(output / "environment.json", hardware)
    write_json(output / "complete.json", dict(recipe_sha256=sha(recipe_path), metrics_sha256=sha(output / "metrics.json")))
    print(json.dumps({"quality": report["metrics"]["quality_all"], "macro_f1": report["macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "e0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--backbone", choices=BACKBONES, default="b0")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    run(args.output, args.device, args.resamples, args.backbone, args.weights)

"""Сверить валидацию с исходными DICOM/Excel и независимо пересчитать срезы OOF.

python -m src.utils.check_validation --output research/results/validation_review.json
Предсказания, пороги, разметка и разбиения не изменяются. GPU не нужен.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pydicom
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .. import config as C
from .prepare_model import digest as sha


def quality(frame, weights=None):
    y, pred, prob = frame.true_quality, frame.pred_quality, frame.prob_quality
    return dict(images=len(frame), studies=int(frame.study.nunique()), positives=int(y.sum()),
                f1=float(f1_score(y, pred, sample_weight=weights)),
                auc=float(roc_auc_score(y, prob, sample_weight=weights)),
                precision=float(precision_score(y, pred, sample_weight=weights, zero_division=0)),
                recall=float(recall_score(y, pred, sample_weight=weights, zero_division=0)))


def review():
    audit = json.loads((C.ARTIFACTS / "audit/summary.json").read_text())
    assert sha(C.LABELS_XLSX) == audit["workbook"]["sha256"]
    for entry in audit["archives"]:
        assert sha(C.RAW / entry["name"]) == entry["sha256"]
    book = pd.read_excel(C.LABELS_XLSX, header=None).iloc[2:]
    assert not book[1].duplicated().any()
    book = book.set_index(1)
    fresh = []
    warnings.filterwarnings("ignore", module="pydicom")
    for path in sorted(C.STUDIES.rglob("*.dcm")):
        ds = pydicom.dcmread(path)
        pixels = ds.pixel_array
        digest = hashlib.sha256(str(pixels.shape).encode() + str(pixels.dtype).encode() + pixels.tobytes()).hexdigest()
        spine = pixels.shape[1] == C.SPINE_COLUMNS
        top, half = pixels[:len(pixels) // 3], pixels.shape[1] // 2
        side = "" if spine else ("left" if top[:, :half].mean() > top[:, half:].mean() else "right")
        columns = [2, 3, 4] if spine else ([5, 6] if side == "right" else [7, 8])
        flags = book.loc[path.relative_to(C.STUDIES).parts[0], columns].to_numpy(dtype=float)
        assert np.isnan(flags).all() or np.isin(flags, [0, 1]).all()
        fresh.append(dict(path_to_study=str(path.relative_to(C.DATA)),
                          study=path.relative_to(C.STUDIES).parts[0], study_uid=str(ds.StudyInstanceUID),
                          anatomical_region=C.REGION_SPINE if spine else C.REGION_FEMUR, side=side,
                          labeled=bool(np.isfinite(flags).all()),
                          patient_id=str(ds.get("PatientID", "")), pixel_sha256=digest, file_sha256=sha(path)))
    files = pd.DataFrame(fresh)
    assert files.groupby("study").study_uid.nunique().eq(1).all()
    assert files.groupby("study_uid").study.nunique().eq(1).all()
    assert files.groupby("pixel_sha256").study.nunique().eq(1).all()
    result = dict(raw_files=len(files), unique_pixels=int(files.pixel_sha256.nunique()),
                  studies=int(files.study.nunique()), patient_id_distinct=int(files.patient_id.nunique()),
                  dicom_study_groups_verified=True, cross_study_exact_duplicates=0,
                  runs=[], independent_test=False, patient_independence_confirmed=False)
    for run in ("e5-blend", "e5-blend-split137"):
        folder = C.ARTIFACTS / run
        manifest = pd.read_csv(folder / "manifest.csv").fillna({"side": ""})
        joined = manifest.merge(files, on="path_to_study", suffixes=("", "_fresh"), validate="one_to_one")
        assert len(joined) == len(manifest) == 249
        for column in ("study", "pixel_sha256", "file_sha256", "anatomical_region", "side"):
            assert joined[column].equals(joined[column + "_fresh"])
        assert set(manifest.pixel_sha256) == set(files.loc[files.labeled, "pixel_sha256"])
        assert not manifest.pixel_sha256.duplicated().any()
        assert manifest.groupby("study").fold.nunique().eq(1).all()
        oof = pd.read_csv(folder / "oof.csv").set_index("image_id", verify_integrity=True)
        assert set(oof.index) == set(manifest.image_id)
        oof = oof.reindex(manifest.image_id).reset_index()
        for row, prediction in zip(manifest.itertuples(), oof.itertuples()):
            spine = row.anatomical_region == C.REGION_SPINE
            columns = [2, 3, 4] if spine else ([5, 6] if row.side == "right" else [7, 8])
            targets = C.TARGETS[:3] if spine else C.TARGETS[3:]
            flags = book.loc[row.study, columns].to_numpy(dtype=float)
            assert np.isin(flags, [0, 1]).all()
            assert row.quality_class == prediction.true_quality == int(flags.max())
            assert prediction.study == row.study and prediction.fold == row.fold
            assert prediction.anatomical_region == row.anatomical_region
            for key, value in zip(targets, flags):
                assert getattr(row, key) == getattr(prediction, "true_" + key) == value
        assert oof.true_quality.isin([0, 1]).all() and oof.pred_quality.isin([0, 1]).all()
        assert oof.prob_quality.between(0, 1).all()
        per_type = {}
        for key, region in zip(C.TARGETS, C.TARGET_REGIONS):
            part = oof[oof.anatomical_region == region]
            per_type[key] = dict(positives=int(part["true_" + key].sum()),
                                f1=float(f1_score(part["true_" + key], part["pred_" + key], zero_division=0)))
        stored = json.loads((folder / "metrics.json").read_text())
        all_quality = quality(oof)
        assert np.isclose(all_quality["f1"], stored["metrics"]["quality_all"]["f1"])
        assert np.isclose(all_quality["auc"], stored["metrics"]["quality_all"]["roc_auc"])
        assert np.isclose(np.mean([v["f1"] for v in per_type.values()]), stored["macro_f1"])
        result["runs"].append(dict(run=run, oof_sha256=sha(folder / "oof.csv"), quality=all_quality,
            macro_f1=stored["macro_f1"], per_type=per_type,
            folds={str(k): quality(g) for k, g in oof.groupby("fold")},
            fold_positive_types={str(k): {t: int(g["true_" + t].sum()) for t in C.TARGETS}
                                 for k, g in oof.groupby("fold")},
            regions={region: quality(g) for region, g in oof.groupby("anatomical_region")},
            equal_study_weight_quality=quality(oof, 1 / oof.groupby("study").study.transform("size")),
            always_positive_f1=float(f1_score(oof.true_quality, np.ones(len(oof)))),
            excluded_unlabeled=int(result["unique_pixels"] - len(oof))))
    result["limitations"] = ["Architecture selected on development OOF; not an unbiased final test",
        "Patient independence and all hip-side assignments cannot be proven from anonymized data",
        "Macro-F1 is the mean of five applicable region/type targets; the official aggregation is not supplied"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "validation-review/independent.json")
    args = parser.parse_args()
    result = review()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

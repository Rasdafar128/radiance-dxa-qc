"""Воспроизвести основной OOF из исходных DICOM через модели внешних фолдов.

python -m src.utils.check_validation_pixels --device cuda
Кэш признаков и финальная модель на всей выборке не используются.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config as C
from ..solution.model import Model


def check(device):
    torch.set_num_threads(2)
    run = C.ARTIFACTS / "e5-blend"
    manifest = pd.read_csv(run / "manifest.csv")
    stored = pd.read_csv(run / "oof.csv").set_index("image_id", verify_integrity=True)
    report = dict(source="raw DICOM through Model.predict", device=device,
                  cached_features_used=False, final_all_data_model_used=False, folds=[])
    for fold in range(3):
        rows = manifest[manifest.fold == fold]
        expected = stored.reindex(rows.image_id)
        model = Model.load(run / f"fold_{fold}", device)
        assert model.metadata["training_partition"] == f"outer_train_{fold}"
        actual = model.predict([C.DATA / p for p in rows.path_to_study], verbose=False)
        assert actual.processing_status.eq("Success").all()
        np.testing.assert_array_equal(actual.anatomical_region, expected.predicted_region)
        np.testing.assert_array_equal(actual.quality_class.to_numpy(dtype=int), expected.pred_quality)
        np.testing.assert_allclose(actual.quality_prob, expected.prob_quality, rtol=0, atol=1e-6)
        for target, region, label in zip(C.TARGETS, C.TARGET_REGIONS, sum(C.VIOLATIONS.values(), [])):
            flags = (actual.anatomical_region == region) & actual.violation_type.map(lambda x: label in x.split(C.VIOLATION_SEP))
            np.testing.assert_array_equal(flags, expected["pred_" + target])
        report["folds"].append(dict(fold=fold, images=len(rows), all_decisions_match=True,
            max_probability_error=float(np.max(np.abs(actual.quality_prob.to_numpy() - expected.prob_quality.to_numpy())))))
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    assert sum(f["images"] for f in report["folds"]) == len(stored) == 249
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "validation-review/raw_pixels.json")
    args = parser.parse_args()
    result = check(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

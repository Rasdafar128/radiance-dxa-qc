"""EfficientNet-B0 и линейные головы — автономный первый рецепт DXA."""

import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pydicom
import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .. import config as C
from .dicom import PREPROCESS, prepare

LOG = logging.getLogger(__name__)
HEADS = ["region", *C.TARGETS, "spine_quality", "hip_quality"]


def report_frame(rows):
    # Nullable integer сохраняет 0/1 и пустую ячейку Failure без значений 0.0/1.0.
    return pd.DataFrame(rows, columns=C.OUTPUT_COLUMNS).astype({"quality_class": "Int64"})


def probability(x, head):
    logits = np.asarray(x, dtype=np.float64) @ np.asarray(head["coef"]) + head["intercept"]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def fit_head(x, y, regularization=0.1, balanced=True):
    # sklearn нужен только для обучения; в поставку идут числовые параметры.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("A head needs both classes in its training partition")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=regularization, class_weight="balanced" if balanced else None,
                                    solver="liblinear", max_iter=2000, random_state=42).fit(scaler.transform(x), y)
    coef = classifier.coef_[0] / scaler.scale_
    intercept = classifier.intercept_[0] - coef @ scaler.mean_
    return dict(coef=coef.tolist(), intercept=float(intercept), threshold=0.5,
                C=regularization, balanced=balanced)


def targets(df):
    y = {"region": (df.anatomical_region == C.REGION_FEMUR).astype(float).to_numpy()}
    for region, key in ((C.REGION_SPINE, "spine"), (C.REGION_FEMUR, "hip")):
        mask = (df.anatomical_region == region).to_numpy()
        y[f"{key}_quality"] = np.where(mask, df.quality_class, np.nan)
    for name, region, label in zip(C.TARGETS, C.TARGET_REGIONS, sum(C.VIOLATIONS.values(), [])):
        applicable = (df.anatomical_region == region) & df.quality_class.notna()
        flags = df.violation_type.fillna("").map(lambda s: label in s.split(C.VIOLATION_SEP))
        y[name] = np.where(applicable, flags.astype(float), np.nan)
    return y


class Model:
    def __init__(self, device="cpu", pretrained=False):
        self.device = torch.device(device)
        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.classifier = torch.nn.Identity()
        self.encoder.to(self.device).eval()
        self.heads = {}
        self.metadata = {"schema": 1, "recipe": "E0-b0-linear-v1", "preprocess": PREPROCESS,
                         "targets": C.TARGETS, "regions": [C.REGION_SPINE, C.REGION_FEMUR]}

    @torch.inference_mode()
    def encode(self, ds):
        tensor = torch.from_numpy(prepare(ds, self.metadata["preprocess"])).unsqueeze(0).to(self.device)
        return self.encoder(tensor).cpu().numpy()[0]

    def features(self, images):
        # ponytail: один кадр на forward гарантирует независимость от состава пакета.
        return np.stack([self.encode(pydicom.dcmread(path)) for path in images])

    def fit(self, df, verbose=True, features=None, parameters=None):
        x = self.features([C.DATA / p for p in df.path_to_study]) if features is None else features
        parameters = parameters or {}
        for key, y in targets(df).items():
            known = np.isfinite(y)
            p = parameters.get(key, {})
            self.heads[key] = fit_head(x[known], y[known], p.get("C", 0.1), p.get("balanced", True))
            self.heads[key]["threshold"] = p.get("threshold", 0.5)
        if verbose:
            LOG.info("Fitted %d heads on %d images", len(self.heads), len(df))
        return self

    def classify(self, x):
        if set(self.heads) != set(HEADS):
            raise ValueError("Model has incomplete heads")
        scores = {key: probability(x, h) for key, h in self.heads.items()}
        regions = np.where(scores["region"] >= 0.5, C.REGION_FEMUR, C.REGION_SPINE)
        rows = []
        for i, region in enumerate(regions):
            violations = [label for name, applicable, label in
                          zip(C.TARGETS, C.TARGET_REGIONS, sum(C.VIOLATIONS.values(), []))
                          if region == applicable and scores[name][i] >= self.heads[name]["threshold"]]
            rows.append(dict(anatomical_region=region, quality_class=int(bool(violations)),
                             quality_prob=float(scores["spine_quality" if region == C.REGION_SPINE else "hip_quality"][i]),
                             violation_type=C.VIOLATION_SEP.join(violations)))
        return pd.DataFrame(rows), scores

    def predict(self, images, verbose=True):
        rows = []
        for source in images:
            started = perf_counter()
            row = dict.fromkeys(C.OUTPUT_COLUMNS, None)
            row.update(path_to_study=str(source), processing_status="Failure")
            try:
                ds = pydicom.dcmread(source)
                row.update(study_uid=str(ds.get("StudyInstanceUID", "")), image_uid=str(ds.get("SOPInstanceUID", "")))
                prediction, _ = self.classify(self.encode(ds)[None])
                row.update(prediction.iloc[0].to_dict(), processing_status="Success")
            except Exception as error:
                if verbose:
                    LOG.warning("Cannot process %s: %s", source, error)
            row["time_of_processing"] = perf_counter() - started
            rows.append(row)
        return report_frame(rows)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}, path / "encoder.pt")
        payload = dict(self.metadata, heads=self.heads,
                       encoder_sha256=hashlib.sha256((path / "encoder.pt").read_bytes()).hexdigest())
        (path / "model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        payload = json.loads((path / "model.json").read_text())
        if payload["schema"] != 1 or payload["targets"] != C.TARGETS or payload["preprocess"] != PREPROCESS:
            raise ValueError("Incompatible model metadata")
        if hashlib.sha256((path / "encoder.pt").read_bytes()).hexdigest() != payload["encoder_sha256"]:
            raise ValueError("Encoder checksum mismatch")
        model = cls(device=device)
        model.encoder.load_state_dict(torch.load(path / "encoder.pt", map_location="cpu", weights_only=True))
        model.heads = payload.pop("heads")
        if set(model.heads) != set(HEADS):
            raise ValueError("Incomplete model artifact")
        for head in model.heads.values():
            if len(head["coef"]) != 1280 or not np.isfinite([*head["coef"], head["intercept"], head["threshold"]]).all():
                raise ValueError("Invalid linear head")
            if not 0 <= head["threshold"] <= 1:
                raise ValueError("Invalid threshold")
        model.metadata = payload
        return model

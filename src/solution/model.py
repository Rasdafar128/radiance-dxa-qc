"""Замороженный визуальный энкодер и линейные головы DXA."""

import hashlib
import json
import logging
import os
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
BACKBONES = ("b0", "dinov2-large", "dinov3-large", "medsiglip", "medimageinsight")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preprocessing(backbone, aspect="pixel_grid"):
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if aspect not in ("pixel_grid", "physical"):
        raise ValueError("Unknown pixel aspect recipe")
    recipe = dict(PREPROCESS, size=224 if backbone == "b0" else 448, aspect=aspect)
    if backbone == "medimageinsight":
        recipe["size"] = 512
    if backbone == "medsiglip":
        recipe.update(mean=[0.5] * 3, std=[0.5] * 3)
    return recipe


def report_frame(rows):
    # Nullable integer сохраняет 0/1 и пустую ячейку Failure без значений 0.0/1.0.
    return pd.DataFrame(rows, columns=C.OUTPUT_COLUMNS).astype({"quality_class": "Int64"})


def probability(x, head):
    logits = np.asarray(x, dtype=np.float64) @ np.asarray(head["coef"]) + head["intercept"]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def fit_head(x, y, regularization=0.1, balanced=True, solver="liblinear"):
    # sklearn нужен только для обучения; в поставку идут числовые параметры.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("A head needs both classes in its training partition")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=regularization, class_weight="balanced" if balanced else None,
                                    solver=solver, max_iter=2000, random_state=42).fit(scaler.transform(x), y)
    coef = classifier.coef_[0] / scaler.scale_
    intercept = classifier.intercept_[0] - coef @ scaler.mean_
    return dict(coef=coef.tolist(), intercept=float(intercept), threshold=0.5,
                C=regularization, balanced=balanced, solver=solver)


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


class MedImageInsight(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        from .medimage_davit import create_encoder

        config = dict(config, SPEC=dict(config["SPEC"], ENABLE_CHECKPOINT=False))
        self.image_encoder = create_encoder(config)
        self.image_projection = torch.nn.Parameter(torch.zeros(2048, 1024))

    def forward(self, tensor):
        x = self.image_encoder.forward_features(tensor) @ self.image_projection
        return torch.nn.functional.normalize(x, dim=-1)


class Model:
    def __init__(self, device="cpu", pretrained=False, backbone="b0", weights=None, encoder_config=None, pooling="global", aspect="pixel_grid"):
        self.device = torch.device(device)
        recipe = preprocessing(backbone, aspect)
        if pooling not in ("global", "spatial") or (pooling == "spatial" and backbone != "dinov3-large"):
            raise ValueError("Spatial pooling is defined only for DINOv3 Large")
        if backbone == "b0":
            self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.encoder.classifier = torch.nn.Identity()
            dimensions = 1280
        elif backbone == "medimageinsight":
            if encoder_config is None:
                encoder_config = json.loads((Path(weights) / "config.json").read_text())
            self.encoder = MedImageInsight(encoder_config)
            if pretrained:
                from safetensors.torch import load_file

                state = load_file(Path(weights) / "original.pt")
                vision = {k: v for k, v in state.items() if k.startswith("image_encoder.") or k == "image_projection"}
                self.encoder.load_state_dict(vision, strict=True)
            dimensions = 1024
        else:
            from transformers import AutoConfig, AutoModel, SiglipVisionConfig, SiglipVisionModel

            if pretrained:
                if weights is None:
                    raise ValueError("A local pretrained snapshot is required")
                encoder_class = SiglipVisionModel if backbone == "medsiglip" else AutoModel
                self.encoder = encoder_class.from_pretrained(weights, local_files_only=True,
                                                             attn_implementation="eager")
            else:
                config = dict(encoder_config)
                model_type = config.pop("model_type")
                if backbone == "medsiglip":
                    self.encoder = SiglipVisionModel(SiglipVisionConfig(**config, attn_implementation="eager"))
                else:
                    self.encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **config),
                                                         attn_implementation="eager")
            expected = {"dinov2-large": "dinov2", "dinov3-large": "dinov3_vit", "medsiglip": "siglip_vision_model"}
            if self.encoder.config.model_type != expected[backbone] or self.encoder.config.hidden_size != (1152 if backbone == "medsiglip" else 1024):
                raise ValueError("Checkpoint does not match the requested backbone")
            dimensions = self.encoder.config.hidden_size
        self.encoder.to(self.device).eval()
        self.heads = {}
        self.metadata = {"schema": 1, "recipe": "E0-b0-linear-v1" if backbone == "b0" else f"frozen-{backbone}-v1",
                         "backbone": backbone, "pooling": pooling,
                         "dimensions": dimensions * (5 if pooling == "spatial" else 1), "preprocess": recipe,
                         "targets": C.TARGETS, "regions": [C.REGION_SPINE, C.REGION_FEMUR]}
        if backbone == "medimageinsight":
            self.metadata["encoder_config"] = encoder_config
        elif backbone != "b0":
            self.metadata["encoder_config"] = self.encoder.config.to_dict()

    @torch.inference_mode()
    def encode(self, ds):
        tensor = torch.from_numpy(prepare(ds, self.metadata["preprocess"])).unsqueeze(0).to(self.device)
        output = self.encoder(tensor)
        backbone = self.metadata.get("backbone", "b0")
        if backbone == "medsiglip":
            output = output.pooler_output
        elif backbone not in ("b0", "medimageinsight"):
            states = output.last_hidden_state
            output = states[:, 0]
            if self.metadata.get("pooling", "global") == "spatial":
                patches = states[:, 1 + self.encoder.config.num_register_tokens:]
                side = int(patches.shape[1] ** 0.5)
                grid = patches.reshape(1, side, side, patches.shape[-1]).permute(0, 3, 1, 2)
                spatial = torch.nn.functional.adaptive_avg_pool2d(grid, (2, 2)).flatten(1)
                output = torch.cat([output, spatial], dim=1)
        return output.cpu().numpy()[0]

    def features(self, images):
        # ponytail: один кадр на forward гарантирует независимость от состава пакета.
        return np.stack([self.encode(pydicom.dcmread(path)) for path in images])

    def fit(self, df, verbose=True, features=None, parameters=None):
        x = self.features([C.DATA / p for p in df.path_to_study]) if features is None else features
        parameters = parameters or {}
        for key, y in targets(df).items():
            known = np.isfinite(y)
            p = parameters.get(key, {})
            self.heads[key] = fit_head(x[known], y[known], p.get("C", 0.1), p.get("balanced", True), p.get("solver", "liblinear"))
            self.heads[key]["threshold"] = p.get("threshold", 0.5)
        if verbose:
            LOG.info("Fitted %d heads on %d images", len(self.heads), len(df))
        return self

    def classify(self, x):
        if set(self.heads) != set(HEADS):
            raise ValueError("Model has incomplete heads")
        scores = {key: probability(x, h) for key, h in self.heads.items()}
        return self.decide(scores)

    def decide(self, scores):
        regions = np.where(scores["region"] >= 0.5, C.REGION_FEMUR, C.REGION_SPINE)
        rows = []
        for i, region in enumerate(regions):
            quality = "spine_quality" if region == C.REGION_SPINE else "hip_quality"
            violations = [label for name, applicable, label in
                          zip(C.TARGETS, C.TARGET_REGIONS, sum(C.VIOLATIONS.values(), []))
                          if region == applicable and scores[name][i] >= self.heads[name]["threshold"]]
            if self.metadata.get("quality_gate", False) and scores[quality][i] < self.heads[quality]["threshold"]:
                violations = []
            rows.append(dict(anatomical_region=region, quality_class=int(bool(violations)),
                             quality_prob=float(scores[quality][i]),
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

    def save(self, path, encoder_path=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / "encoder.tmp"
        if encoder_path is None:
            torch.save({k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}, temporary)
        else:
            # Все головы frozen-опыта используют один неизменный файл весов.
            temporary.unlink(missing_ok=True)
            os.link(encoder_path, temporary)
        temporary.replace(path / "encoder.pt")
        payload = dict(self.metadata, heads=self.heads,
                       encoder_sha256=digest(path / "encoder.pt"))
        (path / "model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        payload = json.loads((path / "model.json").read_text())
        if payload["schema"] == 2 and payload.get("backbone") == "blend":
            if payload["aggregation"] != "mean_probability":
                raise ValueError("Unsupported blend")
            members = [cls.load(path / f"member_{i}", device) for i in range(2)]
            model = Blend(members, payload.pop("heads"), payload["quality_gate"])
            if payload["dimensions"] != model.metadata["dimensions"] or payload["targets"] != C.TARGETS:
                raise ValueError("Incompatible blend metadata")
            model.metadata = payload
            return model
        backbone = payload.get("backbone", "b0")
        aspect = payload["preprocess"]["aspect"]
        if payload["schema"] != 1 or payload["targets"] != C.TARGETS or payload["preprocess"] != preprocessing(backbone, aspect):
            raise ValueError("Incompatible model metadata")
        if digest(path / "encoder.pt") != payload["encoder_sha256"]:
            raise ValueError("Encoder checksum mismatch")
        model = cls(device=device, backbone=backbone, encoder_config=payload.get("encoder_config"),
                    pooling=payload.get("pooling", "global"), aspect=aspect)
        model.encoder.load_state_dict(torch.load(path / "encoder.pt", map_location="cpu", weights_only=True))
        model.heads = payload.pop("heads")
        if set(model.heads) != set(HEADS):
            raise ValueError("Incomplete model artifact")
        for head in model.heads.values():
            if len(head["coef"]) != model.metadata["dimensions"] or not np.isfinite([*head["coef"], head["intercept"], head["threshold"]]).all():
                raise ValueError("Invalid linear head")
            if not 0 <= head["threshold"] <= 1:
                raise ValueError("Invalid threshold")
        model.metadata = payload
        return model


class Blend(Model):
    """Фиксированное среднее двух frozen-моделей; пороги из внутренних OOF."""

    def __init__(self, members, heads, quality_gate=True):
        if len(members) != 2 or set(heads) != set(HEADS):
            raise ValueError("Blend requires two members and complete thresholds")
        if any(not 0 <= h["threshold"] <= 1 for h in heads.values()):
            raise ValueError("Invalid blend threshold")
        self.members, self.heads = members, heads
        self.device = getattr(members[0], "device", torch.device("cpu"))
        self.metadata = dict(schema=2, recipe="frozen-probability-mean-v1", backbone="blend",
                             aggregation="mean_probability", quality_gate=quality_gate,
                             targets=C.TARGETS, dimensions=sum(m.metadata["dimensions"] for m in members),
                             preprocess=members[0].metadata["preprocess"],
                             member_preprocess=[m.metadata["preprocess"] for m in members])

    def encode(self, ds):
        return np.concatenate([m.encode(ds) for m in self.members])

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Use src.utils.train_blend with grouped inner OOF")

    def classify(self, x):
        split = self.members[0].metadata["dimensions"]
        parts = (x[:, :split], x[:, split:])
        scores = {key: np.mean([probability(part, m.heads[key]) for part, m in zip(parts, self.members)], axis=0)
                  for key in HEADS}
        return self.decide(scores)

    def save(self, path, encoder_path=None):
        if encoder_path is not None:
            raise ValueError("Blend stores each member encoder separately")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, member in enumerate(self.members):
            member.save(path / f"member_{i}")
        (path / "model.json").write_text(json.dumps(dict(self.metadata, heads=self.heads),
                                                   ensure_ascii=False, indent=2, allow_nan=False) + "\n")

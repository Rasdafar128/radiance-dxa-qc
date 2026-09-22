"""Frozen foundation-model embeddings (CLS token + mean patch token).

Weights are loaded from ``weights/`` when present (offline container); otherwise downloaded
from the Hugging Face hub (training environment only). Use ``scripts/download_weights.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKBONES = {
    # name: (loader kind, hub id, input size)
    "dinov2_s": ("timm", "vit_small_patch14_dinov2.lvd142m", 518),
    "dinov2_b": ("timm", "vit_base_patch14_dinov2.lvd142m", 518),
    "dinov2_l": ("timm", "vit_large_patch14_dinov2.lvd142m", 518),
    "rad_dino": ("hf", "microsoft/rad-dino", 518),
    "dinov3_l": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m", 448),
    # Microsoft MedImageInsight: public Azure artifact, see scripts/fetch_medimageinsight.py
    "mii": ("mii", "medimageinsight", 512),
}


def pick_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


class Embedder:
    def __init__(self, name: str, device: str | None = None, weights_dir: str | Path | None = None):
        kind, hub_id, size = BACKBONES[name]
        self.name, self.kind, self.size = name, kind, size
        self.device = pick_device(device)
        wdir = Path(weights_dir) if weights_dir else None
        if kind == "timm":
            import timm
            local = wdir / f"{name}.pt" if wdir else None
            if local is not None and local.exists():
                self.model = timm.create_model(hub_id, pretrained=False, num_classes=0)
                self.model.load_state_dict(torch.load(local, map_location="cpu", weights_only=True))
            else:
                self.model = timm.create_model(hub_id, pretrained=True, num_classes=0)
            cfg = self.model.pretrained_cfg
            self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])
        elif kind == "mii":
            import json

            from safetensors.torch import load_file

            from .vendor.medimage_davit import create_encoder

            folder = (wdir / name) if wdir else None
            if folder is None or not folder.exists():
                raise FileNotFoundError("MedImageInsight weights are missing: run scripts/fetch_medimageinsight.py")
            cfg = json.loads((folder / "config.json").read_text())
            cfg = dict(cfg, SPEC=dict(cfg["SPEC"], ENABLE_CHECKPOINT=False))

            class _MII(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.image_encoder = create_encoder(cfg)
                    self.image_projection = torch.nn.Parameter(torch.zeros(2048, 1024))

                def forward(self, x):
                    f = self.image_encoder.forward_features(x) @ self.image_projection
                    return F.normalize(f, dim=-1)

            self.model = _MII()
            state = load_file(folder / "original.pt")
            self.model.load_state_dict({k: v for k, v in state.items()
                                        if k.startswith("image_encoder.") or k == "image_projection"}, strict=True)
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            from transformers import AutoImageProcessor, AutoModel
            local = wdir / name if wdir else None
            src = str(local) if local is not None and local.exists() else hub_id
            self.model = AutoModel.from_pretrained(src)
            proc = AutoImageProcessor.from_pretrained(src)
            self.mean, self.std = tuple(proc.image_mean), tuple(proc.image_std)
        self.model.eval().to(self.device)

    def save(self, weights_dir: str | Path) -> None:
        wdir = Path(weights_dir)
        wdir.mkdir(parents=True, exist_ok=True)
        if self.kind == "timm":
            torch.save(self.model.state_dict(), wdir / f"{self.name}.pt")
        elif self.kind == "mii":
            pass  # already a local artifact fetched by scripts/fetch_medimageinsight.py
        else:
            from transformers import AutoImageProcessor
            self.model.save_pretrained(wdir / self.name)
            AutoImageProcessor.from_pretrained(BACKBONES[self.name][1]).save_pretrained(wdir / self.name)

    def _tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(pad_square(img)).float()[None, None] / 255.0
        return F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)[0]

    @torch.no_grad()
    def __call__(self, imgs: list[np.ndarray], batch: int = 16) -> np.ndarray:
        if not imgs:
            return np.zeros((0, 0), dtype=np.float32)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        feats = []
        for i in range(0, len(imgs), batch):
            t = torch.stack([self._tensor(im) for im in imgs[i:i + batch]]).repeat(1, 3, 1, 1)
            t = ((t - mean) / std).to(self.device)
            if self.kind == "mii":
                feats.append(self.model(t).float().cpu().numpy())
                continue
            if self.kind == "timm":
                tok = self.model.forward_features(t)
                n_prefix = getattr(self.model, "num_prefix_tokens", 1)
                cls, patches = tok[:, 0], tok[:, n_prefix:]
            else:
                out = self.model(pixel_values=t).last_hidden_state
                n_prefix = 1 + getattr(getattr(self.model, "config", None), "num_register_tokens", 0)
                cls, patches = out[:, 0], out[:, n_prefix:]
            feats.append(torch.cat([cls, patches.mean(1)], 1).float().cpu().numpy())
        return np.concatenate(feats)

"""Convert the team's release bundle (encoder .pt + model.json) into the layout dxaqc.embed expects.

The DINOv3-L and MedImageInsight encoders ship as plain ``torch.save`` state dicts next to a
``model.json`` that carries the encoder config. This script materialises

    weights/dinov3_l/   config.json + model.safetensors + preprocessor_config.json  (HF layout)
    weights/mii/        config.json + original.pt (safetensors)                      (vendor layout)

so that ``Embedder("dinov3_l")`` / ``Embedder("mii")`` work fully offline.

python scripts/adapt_release_weights.py --release ../release_weights --out weights
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HF_SKIP = {"return_dict", "output_hidden_states", "torchscript", "pruned_heads", "architectures",
           "transformers_version", "_name_or_path", "dtype", "torch_dtype"}


def build_dinov3(release: Path, out: Path) -> None:
    from transformers import AutoConfig, AutoModel

    meta = json.loads((release / "meta/models/radiance/member_0/model.json").read_text())
    cfg = {k: v for k, v in meta["encoder_config"].items() if k not in HF_SKIP}
    model_type = cfg.pop("model_type")
    config = AutoConfig.for_model(model_type, **cfg)
    model = AutoModel.from_config(config)
    state = torch.load(release / "dinov3-large-encoder.pt", map_location="cpu", weights_only=True)
    expected = set(model.state_dict())
    if not set(state) <= expected:
        # transformers >=5 nests the transformer blocks under `model.`; the release used 4.57.
        state = {(f"model.{k}" if k.startswith("layer.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in state]
    if missing or unexpected:
        raise RuntimeError(f"DINOv3 state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    dst = out / "dinov3_l"
    model.save_pretrained(dst)
    pre = meta["preprocess"]
    (dst / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "BitImageProcessor", "do_resize": True, "do_rescale": True,
        "do_normalize": True, "image_mean": pre["mean"], "image_std": pre["std"],
        "size": {"height": pre["size"], "width": pre["size"]}, "resample": 3,
    }, indent=2) + "\n")
    print("dinov3_l ->", dst, f"({config.hidden_size} dims, image {pre['size']})")


def build_mii(release: Path, out: Path) -> None:
    from safetensors.torch import save_file

    meta = json.loads((release / "meta/models/radiance/member_1/model.json").read_text())
    dst = out / "mii"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(meta["encoder_config"], indent=2) + "\n")
    state = torch.load(release / "medimageinsight-encoder.pt", map_location="cpu", weights_only=True)
    keep = {k: v.contiguous() for k, v in state.items()
            if k.startswith("image_encoder.") or k == "image_projection"}
    if "image_projection" not in keep:
        raise RuntimeError("MedImageInsight projection is missing from the release checkpoint")
    save_file(keep, dst / "original.pt")
    print("mii ->", dst, f"({len(keep)} tensors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="../release_weights")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--only", nargs="*", default=["dinov3_l", "mii"])
    a = ap.parse_args()
    release, out = Path(a.release), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "dinov3_l" in a.only:
        build_dinov3(release, out)
    if "mii" in a.only:
        build_mii(release, out)


if __name__ == "__main__":
    main()

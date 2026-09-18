"""Загрузить фиксированную ревизию HF и сохранить происхождение файлов.

    python -m src.utils.fetch dinov2-large --revision <commit-sha>

Авторизация — через стандартный HF token store; токен не входит в артефакты.
"""

import argparse
import json
import urllib.request

from huggingface_hub import HfApi, snapshot_download

from .. import config as C
from ..solution.model import digest

REPOSITORIES = {
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "medsiglip": "google/medsiglip-448",
}

MII_BASE = "https://hlsmodelonboarding.blob.core.windows.net/models/MedImageInsight/mlflow_model_folder/"
MII_FILES = {
    "original.pt": ("artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt",
                    "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"),
    "config.yaml": ("artifacts/checkpoints/config.yaml",
                    "2b19a6279a6c369b1a9a3eceed905111812ae88961e16ec49748d9318bcc815f"),
    "LICENSE": ("LICENSE", "27ebda9d51f0a56b7e281ccd8230a27236dcb51c05f64b07869ecf6e965d68b0"),
}


def fetch_medimageinsight(destination):
    import yaml

    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in MII_FILES.items():
        path = destination / name
        if not path.exists():
            temporary = path.with_suffix(".download")
            urllib.request.urlretrieve(MII_BASE + url, temporary)
            if digest(temporary) != checksum:
                raise ValueError(f"Official file changed: {name}")
            temporary.replace(path)
        if digest(path) != checksum:
            raise ValueError(f"Official file changed: {name}")
    config = yaml.safe_load((destination / "config.yaml").read_text())["IMAGE_ENCODER"]
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = dict(repository=MII_BASE, revision=MII_FILES["original.pt"][1],
                  sha256={name: digest(destination / name) for name in [*MII_FILES, "config.json"]})
    (destination / "source.json").write_text(json.dumps(source, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=[*REPOSITORIES, "medimageinsight"])
    parser.add_argument("--revision", help="Required immutable HF commit SHA; MII is pinned by SHA256")
    args = parser.parse_args()
    destination = C.ARTIFACTS / "pretrained" / args.backbone
    if args.backbone == "medimageinsight":
        if args.revision:
            parser.error("MedImageInsight uses the pinned official SHA256")
        fetch_medimageinsight(destination)
        print(destination)
        raise SystemExit(0)
    if not args.revision:
        parser.error("HF downloads require --revision")
    name = REPOSITORIES[args.backbone]
    info = HfApi().model_info(name, revision=args.revision)
    if info.sha != args.revision:
        raise ValueError("Specify an immutable commit SHA")
    snapshot_download(name, revision=info.sha, local_dir=destination, max_workers=2,
                      allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors",
                                      "*.safetensors.index.json", "LICENSE*", "README.md"])
    checks = {p.name: digest(p) for p in sorted(destination.iterdir())
              if p.is_file() and p.name != "source.json"}
    source = dict(repository=name, revision=info.sha, sha256=checks)
    (destination / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(destination)

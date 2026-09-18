"""Загрузить фиксированную ревизию HF и сохранить происхождение файлов.

    python -m src.utils.fetch dinov2-large --revision <commit-sha>

Авторизация — через стандартный HF token store; токен не входит в артефакты.
"""

import argparse
import json

from huggingface_hub import HfApi, snapshot_download

from .. import config as C
from ..solution.model import digest

REPOSITORIES = {
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "medsiglip": "google/medsiglip-448",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbone", choices=REPOSITORIES)
    parser.add_argument("--revision", required=True, help="Immutable HF commit SHA")
    args = parser.parse_args()
    name = REPOSITORIES[args.backbone]
    info = HfApi().model_info(name, revision=args.revision)
    if info.sha != args.revision:
        raise ValueError("Specify an immutable commit SHA")
    destination = C.ARTIFACTS / "pretrained" / args.backbone
    snapshot_download(name, revision=info.sha, local_dir=destination, max_workers=2,
                      allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors",
                                      "*.safetensors.index.json", "LICENSE*", "README.md"])
    checks = {p.name: digest(p) for p in sorted(destination.iterdir())
              if p.is_file() and p.name != "source.json"}
    source = dict(repository=name, revision=info.sha, sha256=checks)
    (destination / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(destination)

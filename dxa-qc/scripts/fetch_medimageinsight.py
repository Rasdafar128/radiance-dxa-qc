"""Download the official Microsoft MedImageInsight vision artifact (public Azure blob).

python scripts/fetch_medimageinsight.py [--weights weights]
Checksums are pinned; the .pt file is safetensors (no pickle is executed).
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402

BASE = "https://hlsmodelonboarding.blob.core.windows.net/models/MedImageInsight/mlflow_model_folder/"
FILES = {
    "original.pt": ("artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt",
                    "5eeda63bf616a61664bc95b2c09d3b3d7125209e635678bd3f5f324e9bdb1414"),
    "config.yaml": ("artifacts/checkpoints/config.yaml",
                    "2b19a6279a6c369b1a9a3eceed905111812ae88961e16ec49748d9318bcc815f"),
    "LICENSE": ("LICENSE", "27ebda9d51f0a56b7e281ccd8230a27236dcb51c05f64b07869ecf6e965d68b0"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(config.WEIGHTS_DIR))
    a = ap.parse_args()
    dst = Path(a.weights) / "mii"
    dst.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in FILES.items():
        path = dst / name
        if not path.exists():
            tmp = path.with_suffix(".download")
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(BASE + url, tmp)
            tmp.replace(path)
        got = sha256(path)
        if got != checksum:
            raise SystemExit(f"checksum mismatch for {name}: {got}")
    import yaml
    cfg = yaml.safe_load((dst / "config.yaml").read_text())["IMAGE_ENCODER"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("MedImageInsight ready in", dst)


if __name__ == "__main__":
    main()

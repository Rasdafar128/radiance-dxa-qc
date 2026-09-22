"""Download the public foundation-model weights into ./weights for the offline container.

python scripts/download_weights.py [--weights weights]

DINOv3-L and MedImageInsight are not downloaded here: DINOv3 is licence-gated on the hub and
MedImageInsight ships as an Azure artifact. Both come from the team release bundle — see
scripts/adapt_release_weights.py (and scripts/fetch_medimageinsight.py for the Microsoft artifact).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.embed import Embedder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(config.WEIGHTS_DIR))
a = ap.parse_args()
PUBLIC = {"dinov2_s", "dinov2_b", "dinov2_l", "rad_dino"}
for name in sorted(set(config.EMBEDDERS) | {config.REGION_EMBEDDER}):
    if name not in PUBLIC:
        print("skip", name, "— берётся из релиза, см. scripts/adapt_release_weights.py")
        continue
    Embedder(name, device="cpu").save(a.weights)
    print("saved", name, "->", a.weights)

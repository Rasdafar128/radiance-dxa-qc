#!/usr/bin/env bash
# Build the DXA QC container. Requires trained weights in ./weights (see README, раздел «Обучение»).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${TAG:-dxaqc:1.0}"
for f in weights/qc_model.joblib weights/region_model.joblib weights/dinov2_b.pt \
         weights/rad_dino/config.json weights/dinov3_l/config.json weights/mii/original.pt; do
  [ -e "$f" ] || { echo "missing $f — run scripts/download_weights.py, scripts/adapt_release_weights.py and scripts/train.py first" >&2; exit 1; }
done
docker build --platform linux/amd64 -t "$TAG" .
echo "built $TAG"

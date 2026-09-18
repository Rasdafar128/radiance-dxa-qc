#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mode=${1:-cpu}
case "$mode" in
    cpu) index=cpu ;;
    cuda) index=cu126 ;;
    *) echo 'Usage: docker/run.sh [cpu|cuda]' >&2; exit 2 ;;
esac
python3 -m src.utils.prepare_model
docker build --platform linux/amd64 --build-arg TORCH_INDEX="$index" -f docker/Dockerfile -t "radiance:$mode" .
set -- --rm --platform linux/amd64 --name radiance-api --read-only --tmpfs /tmp:rw,nosuid,size=1g \
    --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8080:8080
if [ "$mode" = cuda ]; then set -- "$@" --gpus all; fi
exec docker run "$@" -e DXA_DEVICE="$mode" "radiance:$mode"

#!/usr/bin/env bash
# Usage:
#   ./run.sh api   [port]                 web UI + REST API on http://localhost:8000
#   ./run.sh batch <input_dir|zip> <output_dir>   offline batch processing -> results.csv/xlsx
set -euo pipefail
TAG="${TAG:-dxaqc:1.0}"
abspath() { mkdir -p "$(dirname "$1")"; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
mode="${1:-api}"
case "$mode" in
  api)
    port="${2:-8000}"
    mkdir -p runtime
    exec docker run --rm -p "${port}:8000" --user "$(id -u):$(id -g)" -v "$(abspath runtime):/data" --name dxaqc "$TAG"
    ;;
  batch)
    [ $# -eq 3 ] || { echo "usage: $0 batch <input> <output_dir>" >&2; exit 2; }
    in="$(abspath "$2")"; out="$(abspath "$3")"; mkdir -p "$out"
    if [ -f "$in" ]; then mount_in="$(dirname "$in")"; arg="/input/$(basename "$in")"; else mount_in="$in"; arg="/input"; fi
    exec docker run --rm --network none --user "$(id -u):$(id -g)" \
      -v "$mount_in:/input:ro" -v "$out:/output" "$TAG" \
      python -m dxaqc.cli --input "$arg" --output-dir /output
    ;;
  *) echo "unknown mode $mode (api|batch)" >&2; exit 2 ;;
esac

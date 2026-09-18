#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test -f artifacts/e0/final/model.json || { echo 'Train or restore artifacts/e0/final first.' >&2; exit 1; }
docker build --platform linux/amd64 -f docker/Dockerfile -t lct26-dxa:e0 .
exec docker run --rm --platform linux/amd64 --name lct26-dxa -p 127.0.0.1:8080:8080 lct26-dxa:e0

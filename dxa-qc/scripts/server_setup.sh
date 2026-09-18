#!/usr/bin/env bash
# One-shot pipeline on the GPU server: env -> weights -> training/validation -> tests -> docker -> smoke test.
#   DATA=/path/to/data ./scripts/server_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-../data}"
python3.11 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu128 || true
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python scripts/download_weights.py
.venv/bin/python scripts/train.py --data "$DATA" --reps 10 --device cuda | tee logs_train.txt
.venv/bin/python -m pytest -q tests
./build.sh
./run.sh batch "$DATA/Для теста" out/smoke
head -5 out/smoke/results.csv
.venv/bin/python scripts/benchmark.py --input "$DATA/Исследования" --device cpu

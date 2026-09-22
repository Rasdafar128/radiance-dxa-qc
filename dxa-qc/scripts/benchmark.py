"""Measure per-study latency and peak memory on the target machine -> reports/performance.md

python scripts/benchmark.py --input /path/to/studies --device cpu
"""
import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.batch import group_studies  # noqa: E402
from dxaqc.pipeline import Engine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="reports/performance.md")
a = ap.parse_args()

t0 = time.perf_counter()
eng = Engine(device=a.device)
load_s = time.perf_counter() - t0
groups = group_studies(Path(a.input))
times, files = [], 0
for key, paths in groups.items():
    t = time.perf_counter()
    rows, _, _ = eng.process_study(key, paths, a.input)
    times.append(time.perf_counter() - t)
    files += len(rows)
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() == "Linux" else 1024 ** 2)
t = np.array(times)
lines = [
    "# Производительность", "",
    f"- машина: {platform.processor() or platform.machine()}, {platform.system()}, device={a.device}",
    f"- загрузка моделей: {load_s:.1f} с",
    f"- исследований: {len(t)}, файлов: {files}",
    f"- время на исследование: среднее {t.mean():.2f} с, медиана {np.median(t):.2f} с, p95 {np.percentile(t, 95):.2f} с, максимум {t.max():.2f} с",
    f"- пиковая RAM процесса: {peak_mb:.0f} МБ",
]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

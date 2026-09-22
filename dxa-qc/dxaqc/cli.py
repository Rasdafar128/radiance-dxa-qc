"""Command line interface.

    python -m dxaqc.cli --input /data/test --output-dir /out [--no-overlays] [--no-sr] [--device cpu]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from . import __version__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dxaqc", description="ИИ-контроль качества DXA-исследований")
    ap.add_argument("--input", required=True, help="папка с DICOM, zip-архив или один .dcm")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weights", default=None, help="папка с весами (по умолчанию ./weights)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-overlays", action="store_true", help="не строить визуализацию")
    ap.add_argument("--no-sr", action="store_true", help="не формировать DICOM SR")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    log = logging.getLogger("dxaqc")
    from .batch import run_batch
    from .pipeline import Engine

    t0 = time.time()
    engine = Engine(a.weights, device=a.device)
    log.info("dxaqc %s, model %s, loaded in %.1fs", __version__, engine.version, time.time() - t0)
    df = run_batch(a.input, a.output_dir, engine, overlays=not a.no_overlays, sr=not a.no_sr,
                   progress=lambda i, n: log.info("study %d/%d", i, n))
    n_fail = int((df.processing_status != "Success").sum())
    log.info("done: %d files, %d failures, %.1fs total -> %s/results.csv", len(df), n_fail,
             time.time() - t0, a.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

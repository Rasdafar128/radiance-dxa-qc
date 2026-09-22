"""Cache embeddings for every unique training image (hips in canonical orientation)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.anatomy import canonical_hip  # noqa: E402
from dxaqc.embed import Embedder  # noqa: E402

d = pd.read_pickle("artifacts/dataset.pkl")
imgs = [canonical_hip(p, s) if r == "hip" else p for p, r, s in zip(d.pixels, d.region, d.side)]
for name in sys.argv[1:]:
    t0 = time.time()
    emb = Embedder(name)
    e = emb(imgs)
    e_flip = emb([np.ascontiguousarray(im[:, ::-1]) for im in imgs])  # spine TTA (tilt sign-invariant)
    np.save(f"artifacts/emb_{name}.npy", e)
    np.save(f"artifacts/emb_{name}_flip.npy", e_flip)
    print(name, e.shape, f"{time.time() - t0:.1f}s")

"""Cache raw+flip embeddings of every available encoder -> artifacts/features_all.pkl"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import config  # noqa: E402
from dxaqc.embed import Embedder  # noqa: E402
from dxaqc.features import geometry_frame  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402

names = sys.argv[1:] or ["dinov2_b", "rad_dino", "dinov2_l", "mii"]
u = build_dataset(Path("../data"))
imgs = list(u.pixels)
flipped = [im[:, ::-1].copy() for im in imgs]
out = Path("artifacts/features_all.pkl")
cache = {"raw": {}, "flp": {}}
if out.exists():
    with open(out, "rb") as f:
        cache = pickle.load(f)
for n in names:
    if n in cache["raw"]:
        print("skip", n, flush=True)
        continue
    e = Embedder(n, weights_dir=config.WEIGHTS_DIR)
    cache["raw"][n] = e(imgs)
    cache["flp"][n] = e(flipped)
    print(n, cache["raw"][n].shape, flush=True)
    del e
cache["geom"], _ = geometry_frame(imgs, u.region.values, u.side.values)
cache["hashes"] = list(u.folder + "/" + u.hash)
out.parent.mkdir(exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(cache, f)
print("saved", out)

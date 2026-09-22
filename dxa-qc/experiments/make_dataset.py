"""Create artifacts/dataset.pkl used by the experiment scripts.

python experiments/make_dataset.py --data ../data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.trainset import build_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data")
a = ap.parse_args()
u = build_dataset(Path(a.data))
Path("artifacts").mkdir(exist_ok=True)
u.to_pickle("artifacts/dataset.pkl")
print(u.groupby(["region", "side", "labeled"]).size())

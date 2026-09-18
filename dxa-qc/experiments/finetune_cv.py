"""CV fine-tuning of a small CNN (same folds as cv_spec.py) -> OOF probabilities.

python scripts/finetune_cv.py --region hip --tasks rot any --arch convnext_nano.in12k_ft_in1k --reps 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.anatomy import canonical_hip  # noqa: E402
from dxaqc.embed import pad_square  # noqa: E402


def prep(img, size):
    t = torch.from_numpy(pad_square(img)).float()[None, None] / 255.0
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]


def augment(x, region, g):
    # x: (1,H,W) in [0,1]; photometric + mild geometric that keeps criteria labels valid
    b, c = (torch.rand(2, generator=g) - 0.5) * torch.tensor([0.2, 0.4])
    gamma = float(torch.exp((torch.rand(1, generator=g) - 0.5) * 0.6))
    x = ((x.clamp(0, 1) ** gamma) - 0.5) * (1 + c) + 0.5 + b
    x = x + torch.randn(x.shape, generator=g) * 0.02
    # small translation (no scaling/cropping: margins matter for ROI)
    s = x.shape[-1]
    dx, dy = (torch.randint(-s // 20, s // 20 + 1, (2,), generator=g)).tolist()
    x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    if region == "spine" and torch.rand(1, generator=g) < 0.5:
        x = torch.flip(x, dims=(2,))  # tilt sign is irrelevant for the axis criterion
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hip")
    ap.add_argument("--tasks", nargs="+", default=["rot", "any"])
    ap.add_argument("--arch", default="convnext_nano.in12k_ft_in1k")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--tag", default="cnn")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    d = pd.read_pickle("artifacts/dataset.pkl")
    m = ((d.region == a.region) & d.labeled).values
    sub = d[m].reset_index(drop=True)
    imgs = [canonical_hip(p, s) if a.region == "hip" else p for p, s in zip(sub.pixels, sub.side)]
    X = torch.stack([prep(im, a.size) for im in imgs])
    Y = torch.tensor(np.stack([sub[f"y_{t}"].values for t in a.tasks], 1), dtype=torch.float32)
    y_any, groups = sub.y_any.values.astype(int), sub.folder.values
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    oof = np.zeros((a.reps, len(sub), len(a.tasks)))
    for r in range(a.reps):
        for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(sub, y_any, groups)):
            t0 = time.time()
            torch.manual_seed(1000 * r + k)
            g = torch.Generator().manual_seed(1000 * r + k)
            model = timm.create_model(a.arch, pretrained=True, num_classes=len(a.tasks), drop_path_rate=0.1).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
            steps = a.epochs * int(np.ceil(len(tr) / a.bs))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
            pos = Y[tr].mean(0).clamp(0.05, 0.95)
            crit = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).to(dev))
            for ep in range(a.epochs):
                model.train()
                perm = torch.randperm(len(tr), generator=g)
                for i in range(0, len(tr), a.bs):
                    idx = tr[perm[i:i + a.bs].numpy()]
                    xb = torch.stack([augment(X[j], a.region, g) for j in idx]).repeat(1, 3, 1, 1)
                    xb = ((xb - mean) / std).to(dev)
                    loss = crit(model(xb), Y[idx].to(dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()
            with torch.no_grad():
                xb = ((X[te].repeat(1, 3, 1, 1) - mean) / std).to(dev)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                if a.region == "spine":
                    p = (p + torch.sigmoid(model(torch.flip(xb, dims=(3,)))).cpu().numpy()) / 2
            oof[r, te] = p
            print(f"rep{r} fold{k} {time.time() - t0:.0f}s", flush=True)
        for j, t in enumerate(a.tasks):
            y = sub[f"y_{t}"].values
            print(f"rep{r} {t} AUC={roc_auc_score(y, oof[r, :, j]):.3f}", flush=True)
    np.save(f"artifacts/oof_{a.tag}_{a.region}.npy", oof)


if __name__ == "__main__":
    main()

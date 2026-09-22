"""Training-set assembly: unique images of the organizers' dataset joined with expert labels.

Label semantics (разметка.xlsx, sheet «Калибровка»): 1 = нарушение, empty = область не исследовалась.
Right/left hip labels are attached to images by anatomy (radiological convention), because
InstanceNumber order and DICOM laterality tags are unreliable in this export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from .anatomy import hip_side_score
from .dicom_io import iter_dicom_files, read_image

LABEL_COLS = ["n", "study", "sp_ukl", "sp_axis", "sp_art", "rh_rot", "rh_roi",
              "lh_rot", "lh_roi", "it_sp", "it_rh", "it_lh", "comment"]


def load_labels(xlsx: Path) -> pd.DataFrame:
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    rows = [list(r[:13]) for r in ws.iter_rows(min_row=3, values_only=True) if r[1]]
    return pd.DataFrame(rows, columns=LABEL_COLS)


def read_training_files(studies_dir: Path) -> pd.DataFrame:
    recs = []
    for study_dir in sorted(p for p in Path(studies_dir).iterdir() if p.is_dir()):
        for p in iter_dicom_files(study_dir):
            r = read_image(p)
            recs.append(dict(folder=study_dir.name, path=str(p), sop=r.sop_uid, study_uid=r.study_uid,
                             inst=r.instance_number, rows=r.rows, cols=r.cols, hash=r.pixel_hash,
                             pixels=r.pixels, error=r.error))
    df = pd.DataFrame(recs)
    bad = df[df.error.notna()]
    if len(bad):
        raise RuntimeError(f"unreadable training files:\n{bad[['path', 'error']]}")
    return df


def _nan(v):
    return np.nan if v is None else float(v)


def build_dataset(data_dir: Path) -> pd.DataFrame:
    """One row per unique image (pixel hash within a study)."""
    data_dir = Path(data_dir)
    files = read_training_files(data_dir / "Исследования")
    labels = load_labels(data_dir / "разметка.xlsx").set_index("study")
    g = files.sort_values(["folder", "inst"]).groupby(["folder", "hash"], sort=False)
    u = g.first().reset_index()
    u["sops"] = g.sop.apply(list).values
    u["paths"] = g.path.apply(list).values
    # Region targets: in this export spine images are 300 px wide and hips 280/248 px; this was
    # verified visually on all 252 images and is used ONLY to create training targets.
    # The service itself predicts the region from pixels (see region model).
    u["region"] = np.where(u.cols == 300, "spine", "hip")
    u["side_score"] = [hip_side_score(p) if r == "hip" else np.nan for p, r in zip(u.pixels, u.region)]
    u["side"] = np.where(u.region == "spine", "", np.where(u.side_score > 0, "R", "L"))
    u["n"] = u.folder.map(labels.n)
    u["comment"] = u.folder.map(labels.comment)

    ys = []
    for row in u.itertuples():
        L = labels.loc[row.folder]
        if row.region == "spine":
            ys.append(dict(y_ukl=_nan(L.sp_ukl), y_axis=_nan(L.sp_axis), y_art=_nan(L.sp_art), y_any=_nan(L.it_sp)))
        elif row.side == "R":
            ys.append(dict(y_rot=_nan(L.rh_rot), y_roi=_nan(L.rh_roi), y_any=_nan(L.it_rh)))
        else:
            ys.append(dict(y_rot=_nan(L.lh_rot), y_roi=_nan(L.lh_roi), y_any=_nan(L.it_lh)))
    u = pd.concat([u, pd.DataFrame(ys, index=u.index)], axis=1)

    dup = u.groupby(["folder", "region", "side"]).size()
    if (dup > 1).any():
        raise RuntimeError(f"several images for one (study, region, side):\n{dup[dup > 1]}")
    # a study row with labels for a region/side that has no image would silently drop labels
    u["labeled"] = u.y_any.notna()
    return u


def label_noise_report(u: pd.DataFrame) -> pd.DataFrame:
    """Images whose 'Итог' disagrees with max(criteria) — reported, used as-is for per-criterion models."""
    rows = []
    for r in u[u.labeled].itertuples():
        crit = [getattr(r, f"y_{t}") for t in (["ukl", "axis", "art"] if r.region == "spine" else ["rot", "roi"])]
        crit = [c for c in crit if not np.isnan(c)]
        if crit and max(crit) != r.y_any:
            rows.append(dict(n=r.n, folder=r.folder, region=r.region, side=r.side,
                             max_criteria=max(crit), itog=r.y_any, comment=r.comment))
    return pd.DataFrame(rows)

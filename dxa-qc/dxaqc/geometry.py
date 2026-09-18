"""Interpretable geometric measurements for DXA quality criteria.

All functions take a uint8 image; hips must already be in canonical orientation
(greater trochanter on the image left, pelvis on the right). Distances are returned in
pixels and in "shaft widths" (anatomical scale, independent of pixel spacing) and mm.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import PIXEL_MM


def _otsu3_low(a: np.ndarray) -> float:
    """Lower threshold of a 3-class Otsu split of a uint8 image (exhaustive over the histogram)."""
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=float)
    w = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    best, t1_best = -1.0, 0
    for t1 in range(1, 254):
        w0, m0 = w[t1], mu[t1]
        if w0 <= 0:
            continue
        t2 = np.arange(t1 + 1, 255)
        w1 = w[t2] - w0
        w2 = 1.0 - w[t2]
        ok = (w1 > 0) & (w2 > 0)
        if not ok.any():
            continue
        m1 = (mu[t2] - m0)[ok] / w1[ok]
        m2 = (mu_t - mu[t2])[ok] / w2[ok]
        var = w0 * (m0 / w0 - mu_t) ** 2 + w1[ok] * (m1 - mu_t) ** 2 + w2[ok] * (m2 - mu_t) ** 2
        k = int(np.argmax(var))
        if var[k] > best:
            best, t1_best = float(var[k]), t1
    return float(t1_best)


def _mask(img: np.ndarray, blur: float = 2.0, low: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Bone mask. ``low=True`` uses the lower multi-Otsu threshold so that the dim cancellous
    trochanteric region is not lost; the caller falls back to plain Otsu if it leaks into soft tissue."""
    a = cv2.GaussianBlur(img, (0, 0), blur)
    if low:
        thr = _otsu3_low(a)
    else:
        thr, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = a > thr
    m = ndi.binary_opening(m, iterations=1)
    return a, m


def _femur(img: np.ndarray, low: bool):
    h, w = img.shape
    a, m = _mask(img, low=low)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    bottom = np.bincount(lab[int(h * 0.9):].ravel(), minlength=n + 1)
    bottom[0] = 0
    if bottom.max() == 0:
        return None
    fem = lab == bottom.argmax()  # femur = largest component touching the bottom 10%
    rows = np.where(fem.any(1))[0]
    y0, y1 = int(rows.min()), int(rows.max())
    xl = np.array([np.argmax(fem[y]) if fem[y].any() else -1 for y in range(h)], float)
    xr = np.array([w - 1 - np.argmax(fem[y, ::-1]) if fem[y].any() else -1 for y in range(h)], float)
    ys = np.arange(int(y1 - 0.3 * (y1 - y0)), y1 + 1)
    ys = ys[xl[ys] >= 0]
    if len(ys) < 5:
        return None
    shaft_w = float(np.median(xr[ys] - xl[ys]))
    if shaft_w < 5:
        return None
    return dict(a=a, m=m, n=n, fem=fem, y0=y0, y1=y1, xl=xl, xr=xr, ys=ys, shaft_w=shaft_w,
                shaft_c=float(np.median((xr[ys] + xl[ys]) / 2)))


# ----------------------------------------------------------------------------- hip
def hip_geometry(img: np.ndarray) -> dict:
    h, w = img.shape
    out = dict(h_px=h, w_px=w, ok=0.0)
    F = _femur(img, low=True)
    F_otsu = _femur(img, low=False)
    # the low threshold may merge the femur with soft tissue: then the shaft looks far too wide
    if F is None or (F_otsu is not None and F["shaft_w"] > 1.6 * F_otsu["shaft_w"]):
        F = F_otsu
    if F is None:
        return out
    a, m, n, fem = F["a"], F["m"], F["n"], F["fem"]
    y0, y1, xl, xr, ys, shaft_w, shaft_c = (F[k] for k in ("y0", "y1", "xl", "xr", "ys", "shaft_w", "shaft_c"))
    pr = np.polyfit(ys, xr[ys], 1)  # medial shaft edge
    pl = np.polyfit(ys, xl[ys], 1)  # lateral shaft edge
    shaft_tilt = float(np.degrees(np.arctan(np.mean([pr[0], pl[0]]))))

    # greater trochanter: lateral-most femur point in the upper part, then climb the lateral
    # contour inside a narrow band until bone ends (does not jump onto the pelvis)
    upper = fem[: int(y0 + 0.65 * (y1 - y0))]
    if not upper.any():
        return out
    lat_x = int(np.where(upper.any(0))[0].min())
    lat_rows = np.where(upper[:, lat_x])[0]
    lat_y = int(lat_rows.mean())
    band = max(3, int(0.6 * shaft_w))
    gt_top = lat_y
    for y in range(lat_y, -1, -1):
        if not fem[y, lat_x:lat_x + band].any():
            break
        gt_top = y
    gt_x = float(np.where(fem[gt_top, lat_x:lat_x + band])[0].mean() + lat_x)

    # lesser trochanter: walk up the medial contour from the bottom while it stays near the shaft;
    # the LT is the most prominent bump before the contour turns into the neck / pelvis
    seg_y, seg_r = [], []
    for y in range(y1, gt_top - 1, -1):
        if xr[y] < 0:
            break
        r = xr[y] - np.polyval(pr, y)
        if r > 1.0 * shaft_w:
            break
        seg_y.append(y)
        seg_r.append(r)
    lt_prom, lt_y, lt_x = 0.0, float("nan"), float("nan")
    if len(seg_r) > 8:
        rs = ndi.uniform_filter1d(np.asarray(seg_r, float), 5)
        after_min = np.minimum.accumulate(rs[::-1])[::-1]  # min of the contour above each row
        bump = rs - after_min
        bump[rs < 0.05 * shaft_w] = 0
        bump[: max(3, int(0.1 * len(rs)))] = 0  # ignore the image bottom (shaft-line fit edge effects)
        k = int(np.argmax(bump))
        if bump[k] > 0.03 * shaft_w:  # otherwise the LT is not visible (over-rotation) -> NaN
            lt_prom, lt_y = float(bump[k]), float(seg_y[k])
            lt_x = float(np.polyval(pr, lt_y) + rs[k])
    medial_resid = np.asarray(seg_r, float) if seg_r else np.zeros(1)
    span = max(y1 - gt_top, 1)
    fem_area = float(fem.sum())
    intensity = a[fem]
    out.update(
        ok=1.0,
        shaft_w_px=shaft_w,
        shaft_c_rel=shaft_c / w,
        shaft_tilt_deg=shaft_tilt,
        gt_top_px=float(gt_top),
        lat_margin_px=float(lat_x),
        bottom_len_px=float(h - gt_top),
        top_margin_sw=gt_top / shaft_w,
        lat_margin_sw=lat_x / shaft_w,
        bottom_len_sw=(h - gt_top) / shaft_w,
        below_lt_sw=(h - lt_y) / shaft_w if np.isfinite(lt_y) else np.nan,
        top_margin_mm=gt_top * PIXEL_MM,
        lat_margin_mm=lat_x * PIXEL_MM,
        below_lt_mm=(h - lt_y) * PIXEL_MM if np.isfinite(lt_y) else np.nan,
        lt_prom_sw=lt_prom / shaft_w,
        lt_rel_y=(lt_y - gt_top) / span if np.isfinite(lt_y) else np.nan,
        medial_resid_max_sw=float(medial_resid.max() / shaft_w),
        fem_area_sw2=fem_area / shaft_w**2,
        fem_frac=fem_area / (h * w),
        bone_frac=float(m.mean()),
        fem_mean_int=float(intensity.mean()) if intensity.size else 0.0,
        img_mean=float(img.mean()),
        aspect=h / w,
        n_components=float(n),
    )
    out["_viz"] = dict(gt_top=(gt_x, float(gt_top)), lat=(float(lat_x), float(lat_y)), lt=(lt_x, lt_y),
                       shaft_l=tuple(map(float, pl)), shaft_r=tuple(map(float, pr)),
                       y_range=(float(y0), float(y1)))
    return out


# --------------------------------------------------------------------------- spine
def spine_geometry(img: np.ndarray) -> dict:
    h, w = img.shape
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    colprof = a[int(h * 0.2):int(h * 0.8)].mean(0)
    x0 = int(np.argmax(np.convolve(colprof, np.ones(70) / 70, "same")))
    half = 45
    xs, ys, ints = [], [], []
    for y in range(int(h * 0.12), int(h * 0.88)):
        lo, hi = max(0, x0 - half), min(w, x0 + half)
        row = a[y, lo:hi]
        row = np.clip(row - np.percentile(row, 20), 0, None)
        if row.sum() < 1:
            continue
        xs.append(lo + (row * np.arange(len(row))).sum() / row.sum())
        ys.append(y)
        ints.append(a[y, lo:hi].mean())
    xs, ys, ints = map(np.asarray, (xs, ys, ints))
    out = dict(h_px=h, w_px=w)
    if len(xs) < 20:
        return out
    wts = np.ones_like(xs)
    for _ in range(5):
        A = np.vstack([ys, np.ones_like(ys)]).T * wts[:, None]
        mslope, c = np.linalg.lstsq(A, xs * wts, rcond=None)[0]
        r = xs - (mslope * ys + c)
        s = np.median(np.abs(r)) * 1.4826 + 1e-6
        wts = 1 / np.maximum(1, np.abs(r) / (2 * s))
    resid = xs - (mslope * ys + c)
    nq = len(ys) // 6
    slope_end = (xs[-nq:].mean() - xs[:nq].mean()) / (ys[-nq:].mean() - ys[:nq].mean())
    quad = np.polyfit(ys, xs, 2)[0] * h**2
    # vertebral period from the intensity profile along the axis
    prof = ints - np.convolve(ints, np.ones(61) / 61, "same")
    prof = prof[30:-30] - prof[30:-30].mean() if len(prof) > 80 else prof - prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-6)
    pitch = float(20 + np.argmax(ac[20:90])) if len(ac) > 90 else float("nan")
    # iliac crests: bright tissue lateral to the spine band in the bottom part of the image
    band_lo, band_hi = max(0, x0 - 55), min(w, x0 + 55)
    lat = a.copy()
    lat[:, band_lo:band_hi] = 0
    lat_bright = lat > np.percentile(a[:, band_lo:band_hi], 60)
    left_rows = lat_bright[:, :band_lo].sum(1) > 6
    right_rows = lat_bright[:, band_hi:].sum(1) > 6
    def first_run_from_bottom(rows_ok):
        idx = np.where(rows_ok[int(h * 0.5):])[0]
        return (h - (int(h * 0.5) + idx.min())) if len(idx) else 0
    crest_l = first_run_from_bottom(left_rows)
    crest_r = first_run_from_bottom(right_rows)
    bottom_lat_frac = float(lat_bright[int(h * 0.85):].mean())
    top_lat_frac = float(lat_bright[: int(h * 0.15)].mean())
    # artefacts: very bright thin/compact structures (metal) anywhere
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    sat = img >= 250
    lab, nsat = ndi.label(sat)
    sizes = np.bincount(lab.ravel())[1:] if nsat else np.array([0])
    p999 = float(np.percentile(img, 99.9))
    spine_p99 = float(np.percentile(img[:, band_lo:band_hi], 99))
    out.update(
        angle_deg=float(np.degrees(np.arctan(abs(mslope)))),
        angle_end_deg=float(np.degrees(np.arctan(abs(slope_end)))),
        dev_max_px=float(np.abs(resid).max()),
        dev_std_px=float(resid.std()),
        curvature=float(abs(quad)),
        x0_rel=x0 / w,
        pitch_px=pitch,
        n_vertebrae=h / pitch if pitch and np.isfinite(pitch) else 0.0,
        crest_h_left=crest_l / h,
        crest_h_right=crest_r / h,
        crest_h_min=min(crest_l, crest_r) / h,
        crest_levels=(max(crest_l, crest_r) / pitch) if pitch else 0.0,
        above_crest_vertebrae=((h - max(crest_l, crest_r)) / pitch) if pitch else 0.0,
        bottom_lat_frac=bottom_lat_frac,
        top_lat_frac=top_lat_frac,
        tophat_p999=float(np.percentile(tophat, 99.9)),
        tophat_top_p99=float(np.percentile(tophat[: int(h * 0.25)], 99)),
        tophat_frac30=float((tophat > 30).mean()),
        sat_frac=float(sat.mean()),
        sat_max_blob=float(sizes.max()),
        n_sat_blobs=float(nsat),
        p999=p999,
        spine_p99=spine_p99,
        bright_ratio=p999 / (spine_p99 + 1),
        img_mean=float(img.mean()),
        aspect=h / w,
    )
    out["_viz"] = dict(axis=(float(mslope), float(c)), y_span=(float(ys.min()), float(ys.max())),
                       crest_rows=(float(h - crest_l) if crest_l else None, float(h - crest_r) if crest_r else None),
                       band=(float(band_lo), float(band_hi)))
    return out

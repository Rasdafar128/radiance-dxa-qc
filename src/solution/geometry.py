"""Image-space geometry from dxa-qc; features only, not clinical measurements."""

import cv2
import numpy as np
from scipy import ndimage as ndi

# Order is part of the saved model contract. Distances stay in pixels.
COLUMNS = ['angle', 'angle_end', 'height', 'lateral', 'top', 'below_lt', 'hip_found']
TASK_COLUMNS = {'spine_axis': [0, 1], 'hip_roi': [2, 3, 4, 5, 6]}


def spine_axis(img):
    h, w = img.shape
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3)
    profile = a[int(h * .2):int(h * .8)].mean(0)
    x0 = int(np.argmax(np.convolve(profile, np.ones(70) / 70, 'same')))
    lo, hi = max(0, x0 - 45), min(w, x0 + 45)
    xs, ys = [], []
    if hi <= lo:
        return [np.nan, np.nan]
    for y in range(int(h * .12), int(h * .88)):
        row = a[y, lo:hi]
        row = np.clip(row - np.percentile(row, 20), 0, None)
        if row.sum() >= 1:
            xs.append(lo + (row * np.arange(len(row))).sum() / row.sum())
            ys.append(y)
    if len(xs) < 20:
        return [np.nan, np.nan]
    xs, ys = np.asarray(xs), np.asarray(ys)
    weights = np.ones_like(xs)
    for _ in range(5):
        design = np.vstack([ys, np.ones_like(ys)]).T * weights[:, None]
        slope, intercept = np.linalg.lstsq(design, xs * weights, rcond=None)[0]
        residual = xs - (slope * ys + intercept)
        scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        weights = 1 / np.maximum(1, np.abs(residual) / (2 * scale))
    n = len(ys) // 6
    end_slope = (xs[-n:].mean() - xs[:n].mean()) / (ys[-n:].mean() - ys[:n].mean())
    return np.degrees(np.arctan(np.abs([slope, end_slope])))


def canonical_hip(img):
    a = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = a > threshold
    h, w = img.shape
    cols = mask[int(h * .8):].sum(0)
    shaft = int((cols * np.arange(w)).sum() / max(cols.sum(), 1.))
    top = (a * mask)[:int(h * .45)]
    return img if top[:, shaft:].sum() > top[:, :shaft].sum() else np.ascontiguousarray(img[:, ::-1])


def _otsu3_low(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    best, threshold = -1., 0
    for t1 in range(1, 254):
        w0, m0 = w[t1], mu[t1]
        if w0 <= 0:
            continue
        t2 = np.arange(t1 + 1, 255)
        w1, w2 = w[t2] - w0, 1 - w[t2]
        ok = (w1 > 0) & (w2 > 0)
        if not ok.any():
            continue
        m1, m2 = (mu[t2] - m0)[ok] / w1[ok], (mu[-1] - mu[t2])[ok] / w2[ok]
        var = w0 * (m0 / w0 - mu[-1])**2 + w1[ok] * (m1 - mu[-1])**2 + w2[ok] * (m2 - mu[-1])**2
        if var.max() > best:
            best, threshold = float(var.max()), t1
    return threshold


def _femur(img, low):
    h, w = img.shape
    a = cv2.GaussianBlur(img, (0, 0), 2.)
    threshold = _otsu3_low(a) if low else cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    labels, n = ndi.label(ndi.binary_opening(a > threshold, iterations=1))
    if not n:
        return None
    bottom = np.bincount(labels[int(h * .9):].ravel(), minlength=n + 1)
    bottom[0] = 0
    if not bottom.max():
        return None
    femur = labels == bottom.argmax()
    rows = np.where(femur.any(1))[0]
    y0, y1 = int(rows.min()), int(rows.max())
    xl = np.array([np.argmax(row) if row.any() else -1 for row in femur], float)
    xr = np.array([w - 1 - np.argmax(row[::-1]) if row.any() else -1 for row in femur], float)
    ys = np.arange(int(y1 - .3 * (y1 - y0)), y1 + 1)
    ys = ys[xl[ys] >= 0]
    if len(ys) < 5:
        return None
    width = float(np.median(xr[ys] - xl[ys]))
    if width < 5:
        return None
    return femur, y0, y1, xl, xr, ys, width


def hip_field(img):
    img = canonical_hip(img)
    h, _ = img.shape
    missing = [h, np.nan, np.nan, np.nan, 0.]
    f, otsu = _femur(img, True), _femur(img, False)
    if f is None or (otsu is not None and f[-1] > 1.6 * otsu[-1]):
        f = otsu
    if f is None:
        return missing
    femur, y0, y1, xl, xr, ys, width = f
    right = np.polyfit(ys, xr[ys], 1)
    upper = femur[:int(y0 + .65 * (y1 - y0))]
    if not upper.any():
        return missing
    lateral = int(np.where(upper.any(0))[0].min())
    lat_y = int(np.where(upper[:, lateral])[0].mean())
    band = max(3, int(.6 * width))
    top = lat_y
    for y in range(lat_y, -1, -1):
        if femur[y, lateral:lateral + band].any():
            top = y
        else:
            break
    seg_y, seg_r = [], []
    for y in range(y1, top - 1, -1):
        if xr[y] < 0:
            break
        r = xr[y] - np.polyval(right, y)
        if r > width:
            break
        seg_y.append(y)
        seg_r.append(r)
    lt_y = np.nan
    if len(seg_r) > 8:
        rs = ndi.uniform_filter1d(np.asarray(seg_r, float), 5)
        bump = rs - np.minimum.accumulate(rs[::-1])[::-1]
        bump[rs < .05 * width] = 0
        bump[:max(3, int(.1 * len(rs)))] = 0
        k = int(np.argmax(bump))
        if bump[k] > .03 * width:
            lt_y = float(seg_y[k])
    return [h, lateral, top, h - lt_y, 1.]


def features(img, is_hip):
    result = np.zeros(len(COLUMNS), dtype=float)
    if min(img.shape) < 70:
        result[TASK_COLUMNS['hip_roi' if is_hip else 'spine_axis']] = np.nan
    elif is_hip:
        result[2:] = hip_field(img)
    else:
        result[:2] = spine_axis(img)
    return result

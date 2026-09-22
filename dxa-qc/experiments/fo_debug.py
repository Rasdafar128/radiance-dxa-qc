"""Diagnostics for the foreign-object highlight: features of every candidate component."""
import sys
from pathlib import Path
import cv2, numpy as np, pydicom
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

img = pydicom.dcmread(sys.argv[1]).pixel_array
th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
core = (th > 40) & (img > 150)
grown = (th > 12) & (img > 90)
lab, _ = ndi.label(grown)
otsu_t, _ = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 2), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bone = cv2.GaussianBlur(img, (0, 0), 2) > otsu_t
lab_b, _ = ndi.label(bone)
sizes_b = np.bincount(lab_b.ravel())
sizes = np.bincount(lab.ravel())
print(f"otsu={otsu_t:.0f}")
for i in sorted(set(np.unique(lab[core])) - {0}):
    comp = lab == i
    ys, xs = np.where(comp)
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    length = max(bh, bw); area = float(sizes[i]); thick = area / length
    ring = ndi.binary_dilation(comp, iterations=5) & ~ndi.binary_dilation(comp, iterations=1)
    bone_around = float((img[ring] > 140).mean())
    enc = set(np.unique(lab_b[comp])) - {0}
    enc_area = max([sizes_b[j] for j in enc], default=0)
    print(f"y={ys.min():3d}-{ys.max():3d} x={xs.min():3d}-{xs.max():3d} area={area:5.0f} len={length:3d} "
          f"thick={thick:4.1f} aspect={length / max(min(bh, bw), 1):4.1f} bone_ring={bone_around:.2f} enclosing_bone={enc_area:6.0f}")

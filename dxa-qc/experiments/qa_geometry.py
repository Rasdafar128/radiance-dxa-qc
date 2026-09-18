"""Visual QA of geometric landmarks on training images -> runs/qa_hips.png, runs/qa_spines.png"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.anatomy import canonical_hip  # noqa: E402
from dxaqc.geometry import hip_geometry, spine_geometry  # noqa: E402
from dxaqc.trainset import build_dataset  # noqa: E402

u = build_dataset(Path(sys.argv[1] if len(sys.argv) > 1 else "../data"))
Path("runs").mkdir(exist_ok=True)


def grid(tiles, cols, path):
    tw = max(t.size[0] for t in tiles); th = max(t.size[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * tw, (k // cols) * th))
    g.save(path)


hips = u[(u.region == "hip") & u.labeled]
sel = list(hips[hips.y_roi == 1].index) + list(hips[hips.y_rot == 1].index[:12]) + list(hips[(hips.y_any == 0)].index[:17])
tiles = []
for i in sel:
    r = u.loc[i]
    img = canonical_hip(r.pixels, r.side)
    g = hip_geometry(img)
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    v = g.get("_viz", {})
    if v:
        gx, gy = v["gt_top"]; d.line([(gx, 0), (gx, gy)], fill=(255, 220, 0)); d.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], outline=(255, 220, 0))
        lx, ly = v["lat"]; d.line([(0, ly), (lx, ly)], fill=(255, 220, 0))
        tx, ty = v["lt"]
        if tx == tx:
            d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], outline=(255, 0, 0), width=2)
    d.rectangle([0, 0, im.size[0], 12], fill=(0, 0, 0))
    d.text((2, 0), f"#{r.n}{r.side} rot{int(r.y_rot)} roi{int(r.y_roi)} top{g.get('top_margin_mm', 0):.0f} lat{g.get('lat_margin_mm', 0):.0f} lt{g.get('below_lt_mm', 0):.0f}", fill=(255, 255, 255))
    tiles.append(im)
grid(tiles, 8, "runs/qa_hips.png")

spines = u[(u.region == "spine") & u.labeled]
sel = list(spines[spines.y_art == 1].index[:8]) + list(spines[spines.y_ukl == 1].index) + list(spines[spines.y_any == 0].index[:10])
tiles = []
for i in sel:
    r = u.loc[i]
    g = spine_geometry(r.pixels)
    im = Image.fromarray(r.pixels).convert("RGB"); d = ImageDraw.Draw(im)
    v = g.get("_viz", {})
    if v:
        m, c = v["axis"]; y0, y1 = v["y_span"]
        d.line([(m * y0 + c, y0), (m * y1 + c, y1)], fill=(255, 0, 0), width=2)
        for cr in v["crest_rows"]:
            if cr is not None:
                d.line([(0, cr), (im.size[0], cr)], fill=(255, 220, 0))
    d.rectangle([0, 0, im.size[0], 12], fill=(0, 0, 0))
    d.text((2, 0), f"#{r.n} ukl{int(r.y_ukl)} ax{int(r.y_axis)} art{int(r.y_art)} ang{g.get('angle_deg', 0):.1f} tophat{g.get('tophat_top_p99', 0):.0f}", fill=(255, 255, 255))
    tiles.append(im)
grid(tiles, 8, "runs/qa_spines.png")
print("ok")

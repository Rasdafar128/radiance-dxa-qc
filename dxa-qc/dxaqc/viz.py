"""Visual explanation: overlay PNG, DICOM Secondary Capture series and a Basic Text SR per study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from . import config

BASIC_TEXT_SR = "1.2.840.10008.5.1.4.1.1.88.11"
SCALE = 3
MAX_ENCLOSING_BONE_PX = 700  # see foreign_object_mask
GREEN, RED, YELLOW, CYAN, WHITE = (60, 220, 90), (240, 60, 60), (250, 210, 60), (60, 200, 240), (255, 255, 255)
STUDY_KEYWORDS = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyInstanceUID", "StudyDate",
                  "StudyTime", "AccessionNumber", "ReferringPhysicianName", "StudyID", "StudyDescription"]


def _font(size: int):
    for p in config.FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(dr, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        probe = f"{cur} {w}".strip()
        if dr.textlength(probe, font=font) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _arrow(dr, p0, p1, color, width=2):
    dr.line([p0, p1], fill=color, width=width)
    for p in (p0, p1):
        dr.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)


def foreign_object_mask(img: np.ndarray, band=(0, 0)) -> np.ndarray:
    """Foreign metal objects (bra wires, clips) by hysteresis on a white top-hat:
    bright thin cores are grown to the whole object at a lower threshold."""
    import cv2
    from scipy import ndimage as ndi
    th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = (th > 40) & (img > 150)
    grown = (th > 12) & (img > 90)
    b0, b1 = int(band[0]), int(band[1])
    core[:, b0:b1] = False  # vertebral processes are bright too: seed only outside the spine
    if not core.any():
        return np.zeros_like(core)
    lab, n = ndi.label(grown)
    sizes = np.bincount(lab.ravel())
    # a foreign object lies in soft tissue, so the bone-threshold blob around it is the object itself;
    # a bright bone edge belongs to a real bone, whose blob is far larger (measured: wires 350-500 px,
    # iliac crest edges 1000-1500 px, lumbar spine ~19000 px)
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab_bone, _ = ndi.label(blur > otsu)
    bone_sizes = np.bincount(lab_bone.ravel())
    keep = []
    for i in set(np.unique(lab[core])) - {0}:
        comp = lab == i
        ys, xs = np.where(comp)
        length = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        thickness = float(sizes[i]) / length
        enclosing = max((bone_sizes[j] for j in set(np.unique(lab_bone[comp])) - {0}), default=0)
        if length >= 12 and thickness <= 8.0 and enclosing <= MAX_ENCLOSING_BONE_PX:
            keep.append(i)
    return np.isin(lab, keep) if keep else np.zeros_like(core)


def render_overlay(img: np.ndarray, res) -> np.ndarray:
    """res: pipeline.ImageResult. Returns an RGB uint8 array with landmarks, measurements and a legend."""
    from scipy import ndimage as ndi
    h, w = img.shape
    base = Image.fromarray(img).convert("RGB").resize((w * SCALE, h * SCALE), Image.BILINEAR)
    n_tasks = len(config.TASKS.get(res.region, []))
    head, foot = 64, 16 * (2 * n_tasks + 2) + 14  # footer: wrapped criterion lines + legend rows
    canvas = Image.new("RGB", (max(base.width, 560), base.height + head + foot), (20, 20, 24))
    canvas.paste(base, (0, head))
    dr = ImageDraw.Draw(canvas)
    f_big, f_small, f_tiny = _font(20), _font(15), _font(13)
    ok = res.quality_class == 0
    side = {"R": " (правое)", "L": " (левое)"}.get(res.side, "")
    dr.text((8, 4), f"{config.REGION_NAMES[res.region]}{side}", fill=WHITE, font=f_small)
    dr.text((8, 26), ("Качественное" if ok else "Нарушение: " + ", ".join(res.violations)) + f"   риск {res.quality_prob:.2f}",
            fill=GREEN if ok else RED, font=f_big)

    def P(x, y):
        return (x * SCALE, y * SCALE + head)

    v, g = res.viz or {}, res.geometry or {}
    flagged = _flagged(res)
    legend = []
    if res.region == "spine" and "axis" in v:
        m, c = v["axis"]
        y0, y1 = v["y_span"]
        col = RED if "axis" in flagged else GREEN
        dr.line([P(m * y0 + c, y0), P(m * y1 + c, y1)], fill=col, width=3)
        xm = m * (y0 + y1) / 2 + c
        dr.line([P(xm, y0), P(xm, y1)], fill=WHITE, width=1)
        dr.text(P(xm + 8, y0 + 4), f"{g.get('angle_deg', 0):.1f}°", fill=col, font=f_big)
        legend += [(col, "линия", "ось позвоночника"), (WHITE, "линия", "вертикаль для отсчёта угла")]
        # iliac crest level: draw only when the detection is plausible (not glued to the image edge)
        for cr in v.get("crest_rows", ()):
            if cr is not None and 0.05 * h < h - cr < 0.5 * h:
                dr.line([P(0, cr), P(w, cr)], fill=YELLOW, width=2)
                dr.text(P(4, cr - 7), "уровень гребней", fill=YELLOW, font=f_tiny)
                legend.append((YELLOW, "линия", "уровень гребней подвздошных костей"))
                break
        if "art" in flagged:
            mask = foreign_object_mask(img, v.get("band", (0, 0)))
            if mask.any():
                tint = np.asarray(canvas).copy()
                ys, xs = np.where(ndi.binary_dilation(mask, iterations=1))
                for x, y in zip(xs, ys):  # translucent cyan fill over the whole object
                    x0, y0_ = x * SCALE, y * SCALE + head
                    blk = tint[y0_:y0_ + SCALE, x0:x0 + SCALE]
                    blk[:] = (0.45 * np.array(CYAN) + 0.55 * blk).astype(np.uint8)
                canvas = Image.fromarray(tint)
                dr = ImageDraw.Draw(canvas)
                legend.append((CYAN, "заливка", "посторонний предмет"))
    elif res.region == "hip" and "gt_top" in v:
        mirror = res.side == "L"

        def X(x):
            return (w - 1 - x) if mirror else x
        col = RED if "roi" in flagged else YELLOW
        gx, gy = v["gt_top"]
        _arrow(dr, P(X(gx), 0), P(X(gx), gy), col)
        dr.text(P(X(gx) + 3, gy / 2), f"{g.get('top_margin_mm', 0):.0f} мм", fill=col, font=f_small)
        lx, ly = v["lat"]
        _arrow(dr, P(X(0), ly), P(X(lx), ly), col)
        dr.text(P(X(lx / 2), ly + 3), f"{g.get('lat_margin_mm', 0):.0f} мм", fill=col, font=f_small)
        legend.append((col, "стрелка", "отступ от кости до края скана"))
        ltx, lty = v["lt"]
        if ltx == ltx and lty == lty:
            rc = RED if "rot" in flagged else GREEN
            dr.ellipse([*P(X(ltx) - 4, lty - 4), *P(X(ltx) + 4, lty + 4)], outline=rc, width=3)
            _arrow(dr, P(X(ltx), lty), P(X(ltx), h - 1), col)
            dr.text(P(X(ltx) + 3, (lty + h) / 2), f"{g.get('below_lt_mm', 0):.0f} мм", fill=col, font=f_small)
            legend.append((rc, "круг", "малый вертел"))
        else:
            legend.append((RED if "rot" in flagged else GREEN, "круг", "малый вертел не визуализируется"))

    # footer: per-criterion verdicts + legend of the drawn elements
    y = canvas.height - foot + 6
    dr.rectangle([0, canvas.height - foot, canvas.width, canvas.height], fill=(20, 20, 24))
    for t, (verdict, why, tail, _rule) in (res.verdicts or {}).items():
        label = {"ukl": "укладка", "axis": "ось", "art": "посторонние предметы",
                 "rot": "укладка/ротация", "roi": "область интереса"}[t]
        bad = verdict != "норма"
        text = f"{label}: {verdict}" + (f" — {why}" if why else "") + f" ({tail})"
        for line in _wrap(dr, text, f_tiny, canvas.width - 16):
            dr.text((8, y), line, fill=RED if bad else GREEN, font=f_tiny)
            y += 16
    x = 8
    for color, kind, text in legend:
        if kind == "заливка":
            dr.rectangle([x, y + 3, x + 16, y + 11], fill=color)
        elif kind == "круг":
            dr.ellipse([x + 3, y + 1, x + 15, y + 13], outline=color, width=2)
        else:
            dr.line([(x, y + 7), (x + 16, y + 7)], fill=color, width=3)
        dr.text((x + 22, y), text, fill=(170, 170, 180), font=f_tiny)
        x += 26 + int(dr.textlength(text, font=f_tiny))
        if x > canvas.width - 180:
            x, y = 8, y + 16
    return np.asarray(canvas)


def _flagged(res) -> set:
    names = set(res.violations)
    return {t for t in config.TASKS[res.region] if config.VIOLATION_NAMES[(res.region, t)] in names}


def _copy_study(ds: Dataset, src: Dataset) -> None:
    for kw in STUDY_KEYWORDS:
        if kw in src:
            ds.add(src.data_element(kw))


def _file_meta(sop_class: str, sop_uid: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return meta


def write_secondary_capture(rgb: np.ndarray, src_path: str, study_key: str, instance_number: int,
                            text: str, out_path: Path) -> Path:
    src = pydicom.dcmread(src_path, stop_before_pixels=True, force=True)
    sop = generate_uid(entropy_srcs=[str(src.get("SOPInstanceUID", src_path)), "dxaqc-sc"])
    ds = Dataset()
    ds.file_meta = _file_meta(SecondaryCaptureImageStorage, sop)
    ds.SpecificCharacterSet = "ISO_IR 192"
    _copy_study(ds, src)
    ds.SOPClassUID, ds.SOPInstanceUID = SecondaryCaptureImageStorage, sop
    ds.Modality, ds.ConversionType = "OT", "WSD"
    ds.SeriesInstanceUID = generate_uid(entropy_srcs=[study_key, "dxaqc-sc-series"])
    ds.SeriesNumber, ds.InstanceNumber = 9901, instance_number
    ds.SeriesDescription = "ИИ контроль качества DXA: визуализация"
    ds.ImageComments = text[:10000]
    ds.BurnedInAnnotation = "YES"
    ref = Dataset()
    ref.ReferencedSOPClassUID = str(src.get("SOPClassUID", ""))
    ref.ReferencedSOPInstanceUID = str(src.get("SOPInstanceUID", ""))
    ds.SourceImageSequence = Sequence([ref])
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.Rows, ds.Columns = int(rgb.shape[0]), int(rgb.shape[1])
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.add_new(0x7FE00010, "OB", np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    c = Dataset()
    c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
    return c


def _text_item(title: str, text: str) -> Dataset:
    it = Dataset()
    it.RelationshipType, it.ValueType = "CONTAINS", "TEXT"
    it.ConceptNameCodeSequence = Sequence([_code("121071", "DCM", "Finding")])
    it.TextValue = f"{title}: {text}"[:10000]
    return it


def write_study_sr(study_key: str, items: list[tuple], out_path: Path) -> Path | None:
    """items: (record, ImageResult) for every analysed file of the study."""
    if not items:
        return None
    src = pydicom.dcmread(items[0][0].path, stop_before_pixels=True, force=True)
    sop = generate_uid(entropy_srcs=[study_key, "dxaqc-sr"])
    ds = Dataset()
    ds.file_meta = _file_meta(BASIC_TEXT_SR, sop)
    ds.SpecificCharacterSet = "ISO_IR 192"
    _copy_study(ds, src)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = BASIC_TEXT_SR, sop, "SR"
    ds.SeriesInstanceUID = generate_uid(entropy_srcs=[study_key, "dxaqc-sr-series"])
    ds.SeriesNumber, ds.InstanceNumber = 9902, 1
    ds.SeriesDescription = "ИИ контроль качества DXA: отчёт"
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    ds.ValueType, ds.ContinuityOfContent = "CONTAINER", "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([_code("DXAQC-REPORT", "99DXAQC", "Контроль качества DXA (ИИ)")])
    ds.ReferencedPerformedProcedureStepSequence = Sequence([])
    ds.PerformedProcedureCodeSequence = Sequence([])
    content, series = [], {}
    seen = set()
    for rec, res in items:
        if rec.pixel_hash in seen:
            continue
        seen.add(rec.pixel_hash)
        side = {"R": " (правое)", "L": " (левое)"}.get(res.side, "")
        verdict = "качественное" if res.quality_class == 0 else "нарушения: " + "; ".join(res.violations)
        content.append(_text_item(f"{config.REGION_NAMES[res.region]}{side}",
                                  f"{verdict}. Риск {res.quality_prob:.2f}. " + " | ".join(res.details)))
        series.setdefault(rec.series_uid, []).append(rec)
    overall = "нарушений не выявлено" if all(r.quality_class == 0 for _, r in items) else "выявлены нарушения качества"
    content.insert(0, _text_item("Итог по исследованию", overall))
    ds.ContentSequence = Sequence(content)
    ev = Dataset()
    ev.StudyInstanceUID = str(src.get("StudyInstanceUID", ""))
    ser_items = []
    for suid, recs in series.items():
        s = Dataset()
        s.SeriesInstanceUID = suid
        refs = []
        for r in recs:
            ref = Dataset()
            ref.ReferencedSOPClassUID = r.meta.get("SOPClassUID", "")
            ref.ReferencedSOPInstanceUID = r.sop_uid
            refs.append(ref)
        s.ReferencedSOPSequence = Sequence(refs)
        ser_items.append(s)
    ev.ReferencedSeriesSequence = Sequence(ser_items)
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([ev])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def save_png(rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path

"""Single source of truth for label vocabularies, specs and runtime settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("DXAQC_WEIGHTS", ROOT / "weights"))

REGION_NAMES = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}

# Official vocabulary, copied verbatim from the organizers' written answers (qa.docx, question 6)
# and from the column headers of разметка.xlsx. The axis label really is spelled «Не выравнена»:
# it is their wording, so it is what the closed test is scored against — do not "fix" it.
VIOLATION_NAMES = {
    ("spine", "ukl"): "Некорректная укладка",
    ("spine", "axis"): "Не выравнена ось позвоночника",
    ("spine", "art"): "Присутствуют посторонние предметы",
    ("hip", "rot"): "Некорректная укладка",
    ("hip", "roi"): "Некорректная область интереса",
}
TASKS = {"spine": ["ukl", "axis", "art"], "hip": ["rot", "roi"]}

# What the criterion requires (ТЗ, раздел 2.3) — shown to the user next to the verdict.
CRITERION_RULES = {
    ("spine", "ukl"): "Снизу в кадре видны верхние края подвздошных костей, сверху — половина позвонка Th12",
    ("spine", "axis"): "Ось позвоночника выровнена, допустимый наклон до 5°",
    ("spine", "art"): "Нет посторонних предметов, выраженных артефактов и наложений (металл от одежды)",
    ("hip", "rot"): "Видны большой вертел, шейка бедра и седалищная кость; ротации нет — малый вертел лишь слегка "
                    "деформирует медиальный контур (не спрятан полностью и не выступает крупно)",
    ("hip", "roi"): "Поле скана: не менее 3 см сверху и снизу от области интереса и не менее 2 см от бокового края",
}
VIOLATION_SEP = ";"

OUTPUT_COLUMNS = [
    "path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
    "violation_type", "processing_status", "time_of_processing",
    # extra columns (allowed by the organizers for quality_prob; others are informative)
    "quality_prob", "side", "details", "error",
]

EMBEDDERS = ["dinov2_b", "rad_dino", "dinov3_l", "mii"]

# Per-task model spec: list of (block kind, block id, L2 strength C).
# Chosen by nested study-grouped CV (see README "Эксперименты").
SPINE_GEOM_UKL = ["crest_levels", "above_crest_vertebrae", "bottom_lat_frac", "crest_h_min"]
SPINE_GEOM_AXIS = ["angle_deg", "angle_end_deg"]
SPINE_GEOM_ART = ["tophat_top_p99", "tophat_p999", "sat_max_blob"]
HIP_GEOM_ROT = ["lt_prom_sw", "shaft_tilt_deg", "medial_resid_max_sw"]
HIP_GEOM_ROI = ["h_px", "lat_margin_mm", "top_margin_mm", "below_lt_mm", "ok"]

SPEC = {
    "ukl": [("geom", SPINE_GEOM_UKL, 1.0), ("emb", "dinov3_l", 0.03), ("emb", "mii", 0.03)],
    "axis": [("geom", SPINE_GEOM_AXIS, 1.0), ("emb", "dinov2_b", 0.03)],
    "art": [("geom", SPINE_GEOM_ART, 1.0), ("emb", "dinov3_l", 0.03), ("emb", "mii", 0.03)],
    "rot": [("emb", "dinov3_l", 0.03)],
    "roi": [("geom", HIP_GEOM_ROI, 1.0), ("emb", "rad_dino", 0.03)],
}
# Direct "is there any violation" head per region. The noisy-OR over criteria is sharp about *which*
# rule is broken, but a head trained on the study-level verdict also sees the cases the expert marked
# as poor without ticking a single criterion. The two are mixed in logit space (ANY_BLEND).
ANY_SPEC = {
    "spine": [("emb", "dinov3_l", 0.03)],
    "hip": [("emb", "dinov3_l", 0.03)],
}
# Measured on repeated study-grouped CV (experiments/any_blend.py): with the old per-criterion
# recipes the direct head was worth +0.04 ROC-AUC, but once ukl/art/rot moved to DINOv3-L and
# MedImageInsight the noisy-OR caught up and the mix stopped paying. Kept as a tunable, off by default.
ANY_BLEND = 0.0

# Contralateral-hip context weight per task (positioning errors are usually bilateral).
CONTEXT_BETA = {"rot": 0.5, "roi": 0.0}

# Exported CR images are resampled to ~isotropic 0.6 mm pixels (vertebral pitch, field size and
# ExposedArea agree; the declared 1.05 mm Y spacing is the native scan pitch). Pending expert reply.
PIXEL_MM = 0.6
MAX_STUDY_SECONDS = 180.0
# Clinical sanity rule: below this measured tilt the axis is considered aligned regardless of the
# model. The TZ tolerance is 5 deg, so 3 deg still leaves 2 deg of margin for measurement error.
# Swept on OOF (experiments/axis_gate.py): axis F1 0.417 -> 0.500, macro-F1 0.591 -> 0.606.
# The gate only filters violation_type — quality_class and its ROC-AUC do not depend on it.
AXIS_MIN_ANGLE_DEG = 3.0
REGION_EMBEDDER = "dinov2_b"
# Hip image with a large saturated (metal-like) area -> implant warning in `details`.
IMPLANT_SAT_FRAC = 0.01
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
SEED = 42

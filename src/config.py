"""Пути, контракт выхода и константы аппарата — единственное, что знают обе половины кода.

Строки нарушений и областей взяты дословно из ответов организаторов (context/organizers/qa.docx):
в выходной таблице они сравниваются как есть, любая переформулировка — это промах
по macro-F1 на всём классе сразу.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TRAIN = DATA / "train"                        # распаковка НД_для_обучения.zip
TEST = DATA / "test"                          # распаковка «Для теста.zip»
ARTIFACTS = ROOT / "artifacts"  # веса
MODEL = ROOT / "models" / "radiance"

STUDIES = TRAIN / "Исследования"
LABELS_XLSX = TRAIN / "разметка.xlsx"

# --- аппарат -----------------------------------------------------------------
# PixelSpacing в DICOM нет; размеры пикселя названы организаторами (context/organizers/qa.docx,
# вопрос 1) и едины для всего набора: один сканер GE Lunar Prodigy Advance.
PIXEL_MM_Y = 1.05
PIXEL_MM_X = 0.60

# Число столбцов однозначно делит области: 300 — позвоночник, иначе бедро.
# Проверено на всех 252 уникальных изображениях обучающего набора и на «Для теста».
SPINE_COLUMNS = 300

# --- контракт выхода ---------------------------------------------------------
REGION_SPINE = "Поясничный отдел позвоночника"
REGION_FEMUR = "Проксимальный отдел бедра"

VIOLATIONS = {
    REGION_SPINE: [
        "Некорректная укладка",
        "Не выравнена ось позвоночника",
        "Присутствуют посторонние предметы",
    ],
    REGION_FEMUR: [
        "Некорректная укладка",
        "Некорректная область интереса",
    ],
}

VIOLATION_SEP = "; "  # несколько нарушений перечисляются через «;», пусто — если их нет

# Стабильные ключи пяти целей; подписи отчёта берём из VIOLATIONS.
TARGETS = ["spine_position", "spine_axis", "spine_objects", "hip_position", "hip_roi"]
TARGET_REGIONS = [REGION_SPINE] * 3 + [REGION_FEMUR] * 2

OUTPUT_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",      # 0 — качественное, 1 — есть нарушение
    "quality_prob",       # [0;1], разрешена организаторами: по 0/1 ROC-AUC вырождается
    "violation_type",
    "processing_status",  # Success / Failure
    "time_of_processing",
]

"""Данные обучения: распаковка архивов, индекс уникальных изображений, разметка → таблица.

    from src.utils import data
    df = data.table()          # строка = изображение, колонки = цель в терминах выхода

Три вещи, которых не видно из кода и которые стоили бы вечера каждому, кто полезет
в набор сам (числа воспроизводит `python -m src.utils.data`):

* Имена внутри zip лежат в cp866 — питон читает их как cp437, и без перекодировки
  получается дерево из «????????????».
* В исследовании 1–29 файлов, но уникальных изображений (по хешу пикселей) ровно
  1–3: остальное — побайтово одинаковые копии в тех же сериях. Между разными
  исследованиями совпадений нет ни одного, поэтому дедуп безопасен, а разбиение по
  исследованию не течёт.
* Какое изображение к какому бедру относится, ни один тег не говорит. Сторона
  берётся из пикселей: таз ярче той половины кадра, к которой повёрнута шейка.
  Внутри всех 74 пар бёдер признак противоположен по знаку (74/74), а знак сверен
  с именами файлов «Для теста» (ППОБ/ЛПОБ) — по одному примеру на сторону.
"""

import hashlib
import warnings
import zipfile

import numpy as np
import pandas as pd
import pydicom

from .. import config as C

# UID'ы после обезличивания не проходят валидацию VR — ругань на каждый файл бесполезна
warnings.filterwarnings("ignore", module="pydicom")

SPINE_COLS = {"укладка": "Некорректная укладка",
              "ось": "Не выравнена ось позвоночника",
              "предметы": "Присутствуют посторонние предметы"}
FEMUR_COLS = {"поз": "Некорректная укладка",
              "roi": "Некорректная область интереса"}  # колонки разметки → строки выхода


def unpack() -> None:
    """Распаковывает архивы организаторов в data/train и data/test (если ещё нет)."""
    for zip_name, target in ((C.RAW / "НД_для_обучения.zip", C.TRAIN),
                             (C.RAW / "Для теста.zip", C.TEST)):
        if target.exists():
            continue
        with zipfile.ZipFile(zip_name) as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                try:  # zipfile декодирует cp866 как cp437, иначе дерево нечитаемо
                    name_ru = name.encode("cp437").decode("cp866")
                except UnicodeError:
                    name_ru = name
                path = target.joinpath(*name_ru.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(name))


def side_score(pixels: np.ndarray) -> float:
    """< 0 — правое бедро, > 0 — левое: в верхней трети кадра таз лежит со стороны шейки."""
    a = pixels.astype(np.float32)
    top = a[: a.shape[0] // 3]
    half = top.shape[1] // 2
    return float(top[:, :half].mean() - top[:, half:].mean()) / (np.ptp(a) + 1e-6)


def index(root=None) -> pd.DataFrame:
    """Уникальные изображения набора: путь, uid'ы, область, сторона."""
    root = C.STUDIES if root is None else root
    rows, seen = [], set()
    for path in sorted(root.rglob("*.dcm")):
        d = pydicom.dcmread(path)
        pixels = d.pixel_array
        digest = hashlib.md5(pixels.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        spine = pixels.shape[1] == C.SPINE_COLUMNS
        score = side_score(pixels)
        rows.append(dict(
            path_to_study=str(path.relative_to(C.DATA)),
            study=path.relative_to(root).parts[0],
            study_uid=str(d.StudyInstanceUID),
            image_uid=str(d.SOPInstanceUID),
            anatomical_region=C.REGION_SPINE if spine else C.REGION_FEMUR,
            side="" if spine else ("left" if score > 0 else "right"),
            height=pixels.shape[0], width=pixels.shape[1], side_score=round(score, 4),
        ))
    return pd.DataFrame(rows)


def labels() -> pd.DataFrame:
    """Разметка организаторов: строка = исследование, колонки = флаги нарушений 0/1."""
    return pd.read_excel(
        C.LABELS_XLSX, header=None, skiprows=2, usecols=range(9),
        names=["n", "study", "укладка", "ось", "предметы",
               "поз_right", "roi_right", "поз_left", "roi_left"],
    ).dropna(subset=["study"])


def table() -> pd.DataFrame:
    """Изображения, соединённые с разметкой: quality_class и violation_type как в выходе."""
    raw = labels()
    flat = []
    for r in raw.itertuples(index=False):
        vals = r._asdict()
        flat.append(dict(study=r.study, anatomical_region=C.REGION_SPINE, side="",
                         violations=[name for key, name in SPINE_COLS.items()
                                     if vals[key] == 1],
                         labeled=not pd.isna(vals["укладка"])))
        for side in ("right", "left"):
            flat.append(dict(study=r.study, anatomical_region=C.REGION_FEMUR, side=side,
                             violations=[name for key, name in FEMUR_COLS.items()
                                         if vals[f"{key}_{side}"] == 1],
                             labeled=not pd.isna(vals[f"поз_{side}"])))
    lab = pd.DataFrame(flat)
    lab = lab[lab.labeled].drop(columns="labeled")

    df = index().merge(lab, on=["study", "anatomical_region", "side"], how="left")
    df["quality_class"] = df.violations.map(
        lambda v: np.nan if not isinstance(v, list) else int(bool(v)))
    df["violation_type"] = df.violations.map(
        lambda v: C.VIOLATION_SEP.join(v) if isinstance(v, list) else "")
    return df.drop(columns="violations")


if __name__ == "__main__":
    unpack()
    df = table()
    print(df.groupby(["anatomical_region", "side"]).quality_class.agg(["size", "sum"]))
    print(df[df.quality_class.notna()].violation_type.value_counts(), "\n")

    assert df.study.nunique() == 100, df.study.nunique()
    assert len(df) == 252, len(df)                       # уникальных изображений
    assert df.quality_class.notna().sum() == 249         # три изображения без разметки
    assert (df.groupby("study").anatomical_region.apply(
        lambda s: (s == C.REGION_SPINE).sum()) <= 1).all()   # позвоночник ≤ 1 на исследование
    assert df[df.side != ""].groupby(["study", "side"]).size().max() == 1  # стороны не дублируются
    print(f"OK: {len(df)} изображений, {df.study.nunique()} исследований, "
          f"{int(df.quality_class.sum())} с нарушением")

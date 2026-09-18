"""Result table writers (csv for automatic scoring, xlsx for people)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=config.OUTPUT_COLUMNS)
    df["quality_class"] = df["quality_class"].astype("Int64")
    df["quality_prob"] = pd.to_numeric(df["quality_prob"], errors="coerce").round(4)
    df["time_of_processing"] = pd.to_numeric(df["time_of_processing"], errors="coerce").astype(float)
    for c in ["path_to_study", "study_uid", "image_uid", "anatomical_region", "violation_type",
              "processing_status", "side", "details", "error"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def write_results(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = out_dir / "results.csv", out_dir / "results.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="results")
        ws = xw.sheets["results"]
        widths = {"A": 60, "B": 45, "C": 45, "D": 32, "E": 14, "F": 55, "G": 18, "H": 18, "I": 12, "J": 6,
                  "K": 90, "L": 40}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
        summary = summarize(df)
        summary.to_excel(xw, index=False, sheet_name="summary")
    return {"csv": csv_path, "xlsx": xlsx_path}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.processing_status == "Success"]
    rows = [("Файлов всего", len(df)), ("Успешно обработано", len(ok)),
            ("Ошибок обработки", int((df.processing_status != "Success").sum())),
            ("Исследований", df.study_uid.nunique()),
            ("Качественных изображений", int((ok.quality_class == 0).sum())),
            ("Изображений с нарушениями", int((ok.quality_class == 1).sum())),
            ("Среднее время на файл, с", round(float(df.time_of_processing.mean()), 3) if len(df) else 0.0)]
    for name in sorted({v for vs in ok.violation_type for v in vs.split(config.VIOLATION_SEP) if v}):
        rows.append((f"Нарушение: {name}", int(ok.violation_type.str.contains(name, regex=False).sum())))
    return pd.DataFrame(rows, columns=["Показатель", "Значение"])

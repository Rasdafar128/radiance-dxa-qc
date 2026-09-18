"""Пакет DICOM — одна строка на файл даже при частичном сбое."""

import argparse
import logging
from pathlib import Path, PurePosixPath
from time import perf_counter
import zipfile

from .. import config as C
from .model import Model, report_frame

MAX_UPLOAD = 256 * 1024**2
MAX_EXPANDED = 512 * 1024**2
MAX_FILE = 32 * 1024**2
MAX_FILES = 10000


def predict_zip(model, source):
    with zipfile.ZipFile(source, metadata_encoding="cp866") as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if not members or len(members) > MAX_FILES:
            raise ValueError("ZIP must contain 1–10000 files")
        if sum(m.file_size for m in members) > MAX_EXPANDED:
            raise ValueError("Expanded ZIP exceeds 512 MiB")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename:
                raise ValueError("Unsafe ZIP member path")
            if member.file_size > MAX_FILE or member.flag_bits & 1:
                raise ValueError("Oversized or encrypted ZIP member")
        rows = []
        for member in members:
            started = perf_counter()
            try:
                # Не распаковываем пути из архива на диск; одинаковые имена не теряются.
                with archive.open(member) as stream:
                    row = model.predict([stream]).iloc[0].to_dict()
            except Exception as error:
                logging.getLogger(__name__).warning("Cannot read ZIP member %s: %s", member.filename, error)
                row = dict.fromkeys(C.OUTPUT_COLUMNS, None)
                row["processing_status"] = "Failure"
            row["path_to_study"] = member.filename
            row["time_of_processing"] = perf_counter() - started
            rows.append(row)
        return report_frame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=C.ARTIFACTS / "e0" / "final")
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    model = Model.load(args.model, args.device)
    if args.input.suffix.lower() == ".zip":
        report = predict_zip(model, args.input)
    else:
        files = sorted(p for p in args.input.rglob("*") if p.is_file()) if args.input.is_dir() else [args.input]
        report = model.predict(files)
    report.to_csv(args.output, index=False)
    print(report.processing_status.value_counts().to_string())

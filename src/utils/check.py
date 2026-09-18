"""Проверка модели — сохранение, независимость пакета, ошибки DICOM/ZIP и API.

    python -m src.utils.check --device cuda
"""

import argparse
import csv
from copy import deepcopy
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import pydicom
import torch
from fastapi.testclient import TestClient

from .. import config as C
from ..solution.batch import predict_zip
from ..solution.dicom import pixels, prepare
from ..solution.model import Model, fit_head, probability
from .evaluate import binary


def zip_bytes(members):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return stream.getvalue()


def check(path, device, output):
    torch.set_num_threads(2)
    started = perf_counter()
    model = Model.load(path, device)
    cold = perf_counter() - started
    files = sorted(C.TEST.rglob("*.dcm"))
    assert len(files) == 3
    ds = pydicom.dcmread(files[0])
    normal = pixels(ds)
    reverse = deepcopy(ds)
    reverse.PhotometricInterpretation = "MONOCHROME1"
    assert np.array_equal(pixels(reverse), 255 - normal)
    recipe = model.metadata["preprocess"]
    assert prepare(ds, recipe).shape == (3, recipe["size"], recipe["size"])
    constant = deepcopy(ds)
    constant.PixelData = np.zeros_like(ds.pixel_array).tobytes()
    try:
        pixels(constant)
        raise AssertionError("Constant image accepted")
    except ValueError:
        pass

    x = np.array([[0, 1], [1, 2], [8, 3], [9, 4]], dtype=float)
    h = fit_head(x, np.array([0, 0, 1, 1]))
    assert (probability(x, h) >= 0.5).tolist() == [False, False, True, True]
    assert binary([0, 0], [0.2, 0.3], [0, 0])["roc_auc"] is None
    assert binary([1, 0, 1], [0.9, 0.1, 0.8], [1, 0, 1])["f1"] == 1

    report = model.predict(files)
    stable_columns = [c for c in C.OUTPUT_COLUMNS if c != "time_of_processing"]
    pd.testing.assert_frame_equal(report[stable_columns], model.predict(files[::-1])[stable_columns].iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(report.iloc[:1][stable_columns], model.predict(files[:1])[stable_columns])
    assert report.processing_status.eq("Success").all()
    assert report.study_uid.iloc[0] == str(ds.StudyInstanceUID)
    with TemporaryDirectory() as temp:
        model.save(temp)
        reloaded = Model.load(temp, device)
        pd.testing.assert_frame_equal(report[stable_columns], reloaded.predict(files)[stable_columns])
        payload = zip_bytes([("good.dcm", files[0].read_bytes()), ("broken.dcm", b"invalid"),
                             ("duplicate.dcm", files[0].read_bytes())])
        mixed = predict_zip(model, BytesIO(payload))
        assert mixed.processing_status.tolist() == ["Success", "Failure", "Success"]
        assert pd.isna(mixed.quality_class.iloc[1]) and pd.isna(mixed.quality_prob.iloc[1])
        assert mixed.quality_prob.iloc[0] == mixed.quality_prob.iloc[2]
        os.environ["DXA_MODEL"], os.environ["DXA_DEVICE"] = temp, device
        from ..solution.api import app
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            response = client.post("/batch", content=payload, headers={"Content-Type": "application/zip"})
            assert response.status_code == 200, response.text
            assert pd.read_csv(BytesIO(response.content)).processing_status.tolist() == ["Success", "Failure", "Success"]
            assert all(r["quality_class"] in ("", "0", "1") for r in csv.DictReader(StringIO(response.text)))
            for invalid in (b"not a zip", zip_bytes([]), zip_bytes([("../escape.dcm", b"x")])):
                assert client.post("/batch", content=invalid, headers={"Content-Type": "application/zip"}).status_code == 400
            assert client.post("/batch", content=payload).status_code == 415
        encoder_file = Path(temp) / ("member_0/encoder.pt" if model.metadata.get("backbone") == "blend" else "encoder.pt")
        with encoder_file.open("ab") as stream:
            stream.write(b"corrupted")
        try:
            Model.load(temp, device)
            raise AssertionError("Corrupted weights accepted")
        except ValueError as error:
            assert "checksum" in str(error)

    all_files = sorted(C.STUDIES.rglob("*.dcm")) + files
    started = perf_counter()
    full = model.predict(all_files)
    elapsed = perf_counter() - started
    assert len(full) == 502 and full.processing_status.eq("Success").all()
    assert full.quality_prob.between(0, 1).all()
    assert (full.quality_class == full.violation_type.ne("").astype(int)).all()
    groups = [str(p.relative_to(C.STUDIES).parts[0]) if p.is_relative_to(C.STUDIES) else "debug" for p in all_files]
    maximum_study = float(full.assign(group=groups).groupby("group").time_of_processing.sum().max())
    assert maximum_study < 180
    candidates = pd.read_csv(C.ARTIFACTS / "audit" / "training_candidates.csv")
    unknown = candidates[candidates.excluded_primary]
    prostheses = model.predict([C.DATA / p for p in unknown.path_to_study])
    assert len(prostheses) == 3 and prostheses.anatomical_region.eq(C.REGION_FEMUR).all()
    assert prostheses.processing_status.eq("Success").all()
    result = dict(images=len(full), success=int(full.processing_status.eq("Success").sum()),
                  seconds=elapsed, cold_load_seconds=cold, max_study_seconds=maximum_study,
                  device=device, checks="MONOCHROME1, save/load, order, singleton, duplicate, partial failure, ZIP/API, prostheses")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checks_{device.replace(':', '_')}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    full.to_csv(output / f"technical_predictions_{device.replace(':', '_')}.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=C.MODEL)
    parser.add_argument("--output", type=Path, default=C.ARTIFACTS / "delivery-checks")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check(args.model, args.device, args.output)

import os
from pathlib import Path

import numpy as np
import pydicom
import pytest

from dxaqc.dicom_io import read_image
from dxaqc.pipeline import ImageResult
from dxaqc.viz import BASIC_TEXT_SR, render_overlay, write_secondary_capture, write_study_sr

ROOT = Path(__file__).resolve().parents[1]


def _res(region="spine"):
    viz = {"axis": (0.05, 140.0), "y_span": (40.0, 270.0), "crest_rows": (280.0, None), "band": (100.0, 200.0)}
    return ImageResult(region=region, region_conf=0.99, ood=False, side="", task_probs={"ukl": 0.1, "axis": 0.8, "art": 0.2},
                       quality_class=1, quality_prob=0.83, violations=["Не выравнена ось позвоночника"],
                       geometry={"angle_deg": 2.9}, viz=viz, details=["наклон оси 2.9°"])


def test_overlay_sc_sr(dicom_factory, tmp_path):
    src = dicom_factory("src.dcm", arr=np.random.default_rng(1).random((315, 300)) * 255)
    rec = read_image(src)
    rgb = render_overlay(rec.pixels, _res())
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    sc = pydicom.dcmread(write_secondary_capture(rgb, str(src), rec.study_uid, 1, "test", tmp_path / "sc.dcm"))
    assert sc.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7" and sc.StudyInstanceUID == rec.study_uid
    assert sc.pixel_array.shape == rgb.shape
    sr = pydicom.dcmread(write_study_sr(rec.study_uid, [(rec, _res())], tmp_path / "sr.dcm"))
    assert sr.SOPClassUID == BASIC_TEXT_SR and len(sr.ContentSequence) == 2


WEIGHTS = ROOT / "weights" / "qc_model.joblib"
TEST_DIR = Path(os.environ.get("DXAQC_TEST_DATA", ROOT.parent / "data" / "Для теста"))


@pytest.mark.skipif(not WEIGHTS.exists() or not TEST_DIR.exists(), reason="needs trained weights and sample data")
def test_end_to_end_on_sample(tmp_path):
    from dxaqc import config
    from dxaqc.batch import run_batch
    from dxaqc.pipeline import Engine

    df = run_batch(TEST_DIR, tmp_path, Engine(device="cpu"))
    assert len(df) == 3 and (df.processing_status == "Success").all()
    assert set(df.anatomical_region) <= set(config.REGION_NAMES.values())
    names = set(config.VIOLATION_NAMES.values())
    for v in df.violation_type:
        assert v == "" or set(v.split(";")) <= names
    assert df.quality_prob.between(0, 1).all()
    # the sample's file names encode the region: ПОП = spine, ППОБ/ЛПОБ = hips
    reg = dict(zip(df.path_to_study, df.anatomical_region))
    assert all(("ПОП" in p) == (r == config.REGION_NAMES["spine"]) for p, r in reg.items())
    assert (tmp_path / "overlays.zip").exists() and (tmp_path / "sr.zip").exists()

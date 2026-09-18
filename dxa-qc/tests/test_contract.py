"""Output contract agreed with the organizers (Разъяснения V2 + typo fix)."""
import numpy as np
import pandas as pd

from dxaqc import config
from dxaqc.model import QualityModel, contralateral
from dxaqc.report import to_frame, write_results


def test_vocabulary_is_exact():
    assert config.REGION_NAMES == {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
    assert set(config.VIOLATION_NAMES.values()) == {
        "Некорректная укладка", "Не выравнена ось позвоночника", "Присутствуют посторонние предметы",
        "Некорректная область интереса"}
    assert config.VIOLATION_SEP == ";"


def test_required_columns_first_and_in_order():
    required = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                "violation_type", "processing_status", "time_of_processing"]
    assert config.OUTPUT_COLUMNS[:8] == required
    assert "quality_prob" in config.OUTPUT_COLUMNS


def _qm():
    return QualityModel(thresholds={
        "spine": {"ukl": 0.5, "axis": 0.5, "art": 0.5, "any": 0.5, "calib": (1.0, 0.0)},
        "hip": {"rot": 0.4, "roi": 0.6, "any": 0.5, "calib": (1.0, 0.0)}})


def test_decide_consistency():
    qm = _qm()
    qc, names, p = qm.decide("spine", {"ukl": 0.1, "axis": 0.2, "art": 0.1, "any": 0.3})
    assert qc == 0 and names == [] and abs(p - 0.3) < 1e-6
    qc, names, _ = qm.decide("spine", {"ukl": 0.9, "axis": 0.7, "art": 0.1, "any": 0.95})
    assert qc == 1 and names == ["Некорректная укладка", "Не выравнена ось позвоночника"]
    qc, names, _ = qm.decide("spine", {"ukl": 0.1, "axis": 0.9, "art": 0.6, "any": 0.95}, blocked={"axis"})
    assert qc == 1 and names == ["Присутствуют посторонние предметы"]
    # positive overall decision but no task above its threshold -> most likely task is reported
    qc, names, _ = qm.decide("hip", {"rot": 0.35, "roi": 0.2, "any": 0.7})
    assert qc == 1 and names == ["Некорректная укладка"]


def test_contralateral_lookup():
    p = np.array([0.9, 0.2, 0.5])
    o = contralateral(p, ["s1", "s1", "s2"], ["R", "L", "R"])
    assert o[0] == 0.2 and o[1] == 0.9 and np.isnan(o[2])


def test_report_types(tmp_path):
    rows = [dict(path_to_study="a.dcm", study_uid="1", image_uid="2", anatomical_region="Проксимальный отдел бедра",
                 quality_class=1, violation_type="Некорректная укладка", processing_status="Success",
                 time_of_processing=0.5, quality_prob=0.81234, side="R", details="", error=None),
            dict(path_to_study="b.dcm", study_uid="1", image_uid="3", anatomical_region="", quality_class=None,
                 violation_type="", processing_status="Failure", time_of_processing=0.1, quality_prob=None,
                 side="", details="", error="invalid DICOM")]
    df = to_frame(rows)
    paths = write_results(df, tmp_path)
    back = pd.read_csv(paths["csv"], keep_default_na=False, dtype=str)
    assert list(back.columns) == config.OUTPUT_COLUMNS
    assert back.quality_class.tolist() == ["1", ""]
    assert back.quality_prob.tolist()[0] == "0.8123"
    assert paths["xlsx"].exists()


def test_lazy_task_embeddings_match_full():
    from dxaqc.features import compute_task_embeddings, task_embeddings

    class FakeEmbedder:
        def __call__(self, imgs):
            return np.stack([np.r_[im.mean(), im[:, :5].mean(), im[:3].mean()] for im in imgs]).astype(np.float32)

    class FakeFX:
        embedders = {n: FakeEmbedder() for n in set(config.EMBEDDERS) | {config.REGION_EMBEDDER}}

    rng = np.random.default_rng(0)
    imgs = [(rng.random((20, 30)) * 255).astype(np.uint8) for _ in range(4)]
    regions, sides = np.array(["spine", "hip", "hip", "spine"]), np.array(["", "R", "L", ""])
    fx = FakeFX()
    raw = {n: fx.embedders[n](imgs) for n in config.EMBEDDERS}
    flp = {n: fx.embedders[n]([np.ascontiguousarray(i[:, ::-1]) for i in imgs]) for n in config.EMBEDDERS}
    full = task_embeddings(raw, flp, regions, sides)
    lazy = compute_task_embeddings(fx, imgs, regions, sides, {config.REGION_EMBEDDER: raw[config.REGION_EMBEDDER]})
    for n in config.EMBEDDERS:
        np.testing.assert_allclose(full[n], lazy[n], rtol=1e-6)

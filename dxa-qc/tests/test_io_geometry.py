import numpy as np

from dxaqc.anatomy import canonical_hip, hip_side_score
from dxaqc.dicom_io import iter_dicom_files, read_image, study_key
from dxaqc.geometry import hip_geometry, spine_geometry


def test_read_uint8(dicom_factory):
    p = dicom_factory()
    r = read_image(p)
    assert r.error is None and r.pixels.dtype == np.uint8 and r.pixels.shape == (300, 280) and r.pixel_hash


def test_read_uint16_monochrome1(dicom_factory):
    arr = np.tile(np.linspace(0, 4000, 280), (300, 1))
    r = read_image(dicom_factory("b.dcm", arr=arr, photometric="MONOCHROME1", bits=16))
    assert r.error is None and r.pixels.dtype == np.uint8
    assert r.pixels[:, 0].mean() > r.pixels[:, -1].mean()  # inverted


def test_bad_file_never_raises(tmp_path):
    p = tmp_path / "broken.dcm"
    p.write_bytes(b"not a dicom at all")
    r = read_image(p)
    assert r.error is not None and r.pixels is None


def test_discovery_and_grouping(dicom_factory, tmp_path):
    a = dicom_factory("s1/x.dcm", study_uid="1.2.3")
    dicom_factory("s1/sub/y", study_uid="1.2.3")  # no extension, DICM preamble
    (tmp_path / "s1" / "notes.txt").write_text("hello")
    files = list(iter_dicom_files(tmp_path))
    assert len(files) == 2 and study_key(a) == "1.2.3"
    linked = tmp_path / "mounted"
    linked.mkdir()
    (linked / "study").symlink_to(tmp_path / "s1", target_is_directory=True)
    assert len(list(iter_dicom_files(linked))) == 2  # symlinked study folders are followed


def _synthetic_spine(angle_deg=6.0, h=315, w=300):
    img = np.zeros((h, w), np.float32)
    ys = np.arange(h)
    cx = 150 + np.tan(np.radians(angle_deg)) * (ys - h / 2)
    xx = np.arange(w)[None, :]
    band = np.abs(xx - cx[:, None]) < 30
    img[band] = 150
    img += (40 * (np.sin(ys / 50 * 2 * np.pi) > 0))[:, None] * band
    return np.clip(img, 0, 255).astype(np.uint8)


def test_spine_axis_angle():
    g = spine_geometry(_synthetic_spine(6.0))
    assert 3.5 < g["angle_deg"] < 8.5
    g0 = spine_geometry(_synthetic_spine(0.0))
    assert g0["angle_deg"] < 1.5


def _synthetic_right_hip(h=260, w=280):
    img = np.zeros((h, w), np.uint8)
    img[110:, 90:140] = 170                       # shaft
    img[60:130, 60:150] = 170                      # trochanteric region (lateral = image left)
    img[20:110, 180:270] = 200                     # pelvis / acetabulum (medial = image right)
    img[70:100, 140:185] = 170                     # neck
    return img


def test_hip_side_and_geometry():
    right = _synthetic_right_hip()
    assert hip_side_score(right) > 0
    assert hip_side_score(right[:, ::-1]) < 0
    g = hip_geometry(canonical_hip(right, "R"))
    assert g["ok"] == 1.0 and 40 < g["shaft_w_px"] < 60 and g["lat_margin_px"] <= 62


def test_zip_rejects_sibling_prefix(tmp_path):
    import zipfile
    import pytest
    from dxaqc.dicom_io import extract_zip

    source = tmp_path / 'input.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('../out-escape/file.dcm', b'outside')
    with pytest.raises(ValueError):
        extract_zip(source, tmp_path / 'out')
    assert not (tmp_path / 'out-escape').exists()

import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CR = "1.2.840.10008.5.1.4.1.1.1"


def make_dicom(path: Path, arr: np.ndarray, study_uid: str | None = None, photometric="MONOCHROME2", bits=8):
    meta = FileMetaDataset()
    sop = generate_uid()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = CR, sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = CR, sop
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality, ds.InstanceNumber, ds.PatientName = "CR", 1, "Anonymized"
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated = ds.BitsStored = bits
    ds.HighBit, ds.PixelRepresentation = bits - 1, 0
    dtype = np.uint8 if bits == 8 else np.uint16
    ds.PixelData = arr.astype(dtype).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def dicom_factory(tmp_path):
    def _f(name="a.dcm", arr=None, **kw):
        arr = arr if arr is not None else (np.random.default_rng(0).random((300, 280)) * 255)
        return make_dicom(tmp_path / name, arr, **kw)
    return _f

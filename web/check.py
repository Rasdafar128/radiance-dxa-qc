"""Проверка границы веб-шлюза без GPU: python -m web.check."""

import io
import base64
import struct
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from .app import app
from .previews import build_previews
from .annotations import annotate
from src.solution.geometry import spine_axis, hip_field
from src.config import REGION_SPINE, REGION_FEMUR


CSV = ("path_to_study,study_uid,image_uid,anatomical_region,quality_class,quality_prob,"
       "violation_type,processing_status,time_of_processing\n"
       '"<script>.dcm",1,2,region,0,0.1,,Success,0.3\n')


def dicom_bytes(inverse=False):
    ds = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b"\0" * 128)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = 4, 4
    ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 1, 8, 8, 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME1" if inverse else "MONOCHROME2"
    ds.PixelData = bytes(range(0, 256, 16))
    output = io.BytesIO()
    ds.save_as(output)
    return output.getvalue()


class PreviewCheck(unittest.TestCase):
    def test_duplicate_names_keep_distinct_pixels_and_monochrome1(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("a/scan.dcm", dicom_bytes())
            with self.assertWarns(UserWarning):
                archive.writestr("a/scan.dcm", dicom_bytes(inverse=True))
            archive.writestr("broken.dcm", b"broken")
        rows = [{"path_to_study": p, "processing_status": "Success"}
                for p in ("a/scan.dcm", "a/scan.dcm", "broken.dcm")]
        previews = build_previews(source, rows)
        decoded = [np.array(Image.open(io.BytesIO(base64.b64decode(p["image"].split(",")[1]))))
                   for p in previews[:2]]
        np.testing.assert_array_equal(decoded[0], np.arange(0, 256, 16).reshape(4, 4))
        np.testing.assert_array_equal(decoded[1], 255 - decoded[0])
        self.assertNotIn("image", previews[2])
        self.assertEqual(previews[0]["width"], 4)
        self.assertFalse(previews[0]["reduced"])
        rows[0]["path_to_study"] = "wrong-file.dcm"
        self.assertTrue(all("image" not in p for p in build_previews(source, rows)))

    def test_annotation_failure_preserves_original_preview(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("scan.dcm", dicom_bytes())
        rows = [{"path_to_study": "scan.dcm", "processing_status": "Success"}]
        with patch("web.previews.annotate", side_effect=ValueError("missing landmarks")):
            preview = build_previews(source, rows)[0]
        self.assertIn("image", preview)
        self.assertNotIn("overlay", preview)
        self.assertIn("недоступна", preview["annotation_notes"][0])

    def test_overlay_budget_preserves_source_image(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("scan.dcm", dicom_bytes())
        rows = [{"path_to_study": "scan.dcm", "processing_status": "Success"}]
        image = build_previews(source, rows)[0]["image"]
        budget = len(base64.b64decode(image.split(',')[1]))
        with patch("web.previews.MAX_PREVIEWS", budget), patch("web.previews.annotate",
                return_value=(Image.new('RGBA',(4,4),'orange'),[],[])):
            preview = build_previews(source, rows)[0]
        self.assertEqual(preview['image'], image)
        self.assertNotIn('overlay', preview)
        self.assertIn('Лимит разметки',preview['annotation_notes'][0])

    def test_preview_limits_do_not_remove_results(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("scan.dcm", dicom_bytes())
        rows = [{"path_to_study": "scan.dcm", "processing_status": "Success"}]
        with patch("web.previews.MAX_PREVIEWS", 1):
            self.assertEqual(len(build_previews(source, rows)), 1)
            self.assertIn("Лимит", build_previews(source, rows)[0]["message"])


class AnnotationCheck(unittest.TestCase):
    def test_axis_overlay_preserves_features_and_pixels(self):
        y, x = np.mgrid[:315, :300]
        img = (np.abs(x - (150 + .1 * (y - 157))) < 30).astype(np.uint8) * 170
        original = img.copy()
        landmarks = {}
        np.testing.assert_array_equal(spine_axis(img), spine_axis(img, landmarks))
        overlay, legend, _ = annotate(img, {"anatomical_region": REGION_SPINE,
            "violation_type": "Не выравнена ось позвоночника"})
        self.assertEqual(overlay.size, (300, 315))
        self.assertTrue(overlay.getbbox())
        self.assertTrue(legend[0]["flagged"])
        np.testing.assert_array_equal(img, original)
        empty, labels, _ = annotate(img, {"anatomical_region": "unknown"})
        self.assertIsNone(empty.getbbox())
        self.assertEqual(labels, [])

    def test_missing_and_implausible_crests_have_no_legend(self):
        y, x = np.mgrid[:300, :300]
        for edge in (None, 140, 290):
            img = (np.abs(x-150) < 25).astype(np.uint8)*170
            if edge is not None:
                img[edge:, :65] = 200
                img[edge:, 235:] = 200
            overlay, legend, notes = annotate(img, {"anatomical_region": REGION_SPINE,
                "violation_type": "Некорректная укладка"})
            self.assertFalse(any('боковые' in item['label'] for item in legend))
            self.assertEqual(notes.count('Нижние боковые ориентиры не найдены.'), 1)
        _, _, notes = annotate(np.zeros((300,300),dtype=np.uint8),
                                {"anatomical_region": REGION_FEMUR, "violation_type": ""})
        self.assertNotIn('нарушения', ' '.join(notes))

    def test_hip_mirrors_back_to_original_coordinates(self):
        y, x = np.mgrid[:290, :280]
        mask = ((x > 70) & (x < 115) & (y > 100)) | (((x-120)/65)**2 + ((y-85)/50)**2 < 1)
        img = mask.astype(np.uint8)*180
        row = {"anatomical_region": REGION_FEMUR, "violation_type": "Некорректная область интереса"}
        landmarks = {}
        np.testing.assert_array_equal(hip_field(img), hip_field(img, landmarks))
        left, _, _ = annotate(img, row)
        right, _, _ = annotate(np.ascontiguousarray(img[:, ::-1]), row)
        self.assertTrue(left.getbbox())
        np.testing.assert_array_equal(np.asarray(left)[:, ::-1], np.asarray(right))


class GatewayCheck(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()
        self.uploads = []

        def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "model": "Radiance", "version": "1.0",
                                                  "backbones": ["dinov3-large", "medimageinsight"]})
            self.uploads.append(request.content)
            return httpx.Response(200, text=CSV, headers={"content-type": "text/csv"})

        self.mock = httpx.AsyncClient(base_url="http://model", transport=httpx.MockTransport(upstream))
        self.real = app.state.client
        app.state.client = self.mock

    def tearDown(self):
        self.client.portal.call(self.mock.aclose)
        app.state.client = self.real
        self.client.__exit__(None, None, None)

    def test_dicom_wrapper_and_exact_csv(self):
        response = self.client.post("/api/analyze?filename=../../image.dcm", content=b"dicom",
                                    headers={"Content-Type": "application/dicom"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["csv"], CSV)
        self.assertEqual(response.json()["model"], "Radiance")
        self.assertEqual(response.json()["rows"][0]["path_to_study"], "<script>.dcm")
        with zipfile.ZipFile(io.BytesIO(self.uploads[0])) as archive:
            self.assertEqual(archive.namelist(), ["image.dcm"])
            self.assertEqual(archive.read("image.dcm"), b"dicom")
        self.assertFalse(app.state.lock.locked())

    def test_real_preview_and_csv_are_returned_together(self):
        response = self.client.post("/api/analyze?filename=%3Cscript%3E.dcm", content=dicom_bytes(),
                                    headers={"Content-Type": "application/dicom"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["csv"], CSV)
        self.assertTrue(response.json()["previews"][0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(response.json()["previews"][0]["height"], 4)

    def test_limits_and_empty_input(self):
        headers = {"Content-Type": "application/zip"}
        with patch("web.app.MAX_UPLOAD", 4):
            self.assertEqual(self.client.post("/api/analyze", content=b"12345", headers=headers).status_code, 413)
        self.assertEqual(self.client.post("/api/analyze", content=b"", headers=headers).status_code, 400)
        self.assertEqual(self.client.post("/api/analyze", content=b"123").status_code, 415)
        self.assertEqual(self.uploads, [])
        self.assertFalse(app.state.lock.locked())

    def test_zip_is_forwarded_unchanged(self):
        response = self.client.post("/api/analyze", content=b"zip-bytes", headers={"Content-Type": "application/zip"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.uploads, [b"zip-bytes"])

    def test_retry_selects_duplicate_by_position_and_rejects_invalid_index(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("folder/", b"")
            archive.writestr("folder/scan.dcm", b"first")
            with self.assertWarns(UserWarning):
                archive.writestr("folder/scan.dcm", b"second")
        headers = {"Content-Type": "application/zip"}
        response = self.client.post("/api/analyze?image_index=1", content=source.getvalue(), headers=headers)
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(self.uploads[0])) as archive:
            self.assertEqual(archive.namelist(), ["folder/scan.dcm"])
            self.assertEqual(archive.read("folder/scan.dcm"), b"second")
        for index, status in [(2, 400), (-1, 422)]:
            self.assertEqual(self.client.post(f"/api/analyze?image_index={index}",
                                             content=source.getvalue(), headers=headers).status_code, status)
        self.assertEqual(len(self.uploads), 1)
        self.assertFalse(app.state.lock.locked())

    def test_retry_rejects_corrupted_compressed_member(self):
        for compression, offset, invalid in [(zipfile.ZIP_DEFLATED, 0, 7), (zipfile.ZIP_LZMA, 4, 255)]:
            with self.subTest(compression=compression):
                source = io.BytesIO()
                with zipfile.ZipFile(source, "w", compression=compression) as archive:
                    archive.writestr("scan.dcm", b"dicom" * 20)
                payload = bytearray(source.getvalue())
                name_size, extra_size = struct.unpack_from("<HH", payload, 26)
                # Повреждаем блок DEFLATE или свойства LZMA, сохраняя структуру ZIP.
                payload[30 + name_size + extra_size + offset] = invalid
                response = self.client.post("/api/analyze?image_index=0", content=bytes(payload),
                                            headers={"Content-Type": "application/zip"})
                self.assertEqual(response.status_code, 400)
                self.assertIn("Не удалось повторить", response.json()["detail"])
                self.assertEqual(self.uploads, [])
                self.assertFalse(app.state.lock.locked())

    def test_busy_gateway(self):
        self.client.portal.call(app.state.lock.acquire)
        try:
            response = self.client.post("/api/analyze", content=b"zip", headers={"Content-Type": "application/zip"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(self.uploads, [])
        finally:
            app.state.lock.release()

    def test_wrong_model_is_not_used(self):
        self.mock._transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "status": "ok", "model": "other", "version": "1.0",
            "backbones": ["dinov3-large", "medimageinsight"]}))
        response = self.client.post("/api/analyze", content=b"zip", headers={"Content-Type": "application/zip"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.uploads, [])
        self.assertFalse(app.state.lock.locked())

    def test_upstream_failure_and_static_boundary(self):
        def unavailable(request):
            raise httpx.ConnectError("private endpoint details", request=request)
        self.mock._transport = httpx.MockTransport(unavailable)
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private endpoint", response.text)
        self.assertEqual(self.client.get("/static/../app.py").status_code, 404)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()

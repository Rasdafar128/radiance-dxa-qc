"""Проверка границы веб-шлюза без GPU: python -m web.check."""

import io
import base64
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

    def test_preview_limits_do_not_remove_results(self):
        source = io.BytesIO()
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("scan.dcm", dicom_bytes())
        rows = [{"path_to_study": "scan.dcm", "processing_status": "Success"}]
        with patch("web.previews.MAX_PREVIEWS", 1):
            self.assertEqual(len(build_previews(source, rows)), 1)
            self.assertIn("Лимит", build_previews(source, rows)[0]["message"])


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

    def test_segmentation_unavailable_and_bad_input(self):
        prompt = {"image": "data:image/png;base64,AA==", "box": [0, 0, 1, 1]}
        self.assertEqual(self.client.post('/api/segment', json=prompt).status_code, 503)
        prompt['box'] = [1, 0, 0, 1]
        self.assertEqual(self.client.post('/api/segment', json=prompt).status_code, 400)
        with patch('web.app.MAX_REQUEST', 8):
            self.assertEqual(self.client.post('/api/segment', json=prompt).status_code, 413)
        self.assertFalse(app.state.lock.locked())

    def test_segmentation_proxy_preserves_mask(self):
        from src.solution.segmentation import png_data
        mask = png_data(np.array([[0, 255], [255, 0]], dtype=np.uint8))
        expected = dict(model='Radiance Anatomy', mask=mask, overlay=mask, width=2, height=2, empty=False)
        prompt = dict(image=mask, box=[0, 0, 1, 1])
        calls = []
        def upstream(request):
            if request.url.path == '/health':
                return httpx.Response(200, json=dict(status='ok', model='Radiance', version='1.0',
                    backbones=['dinov3-large', 'medimageinsight'], segmentation=True))
            calls.append(request)
            return httpx.Response(200, json=expected)
        self.mock._transport = httpx.MockTransport(upstream)
        self.assertEqual(self.client.post('/api/segment', json=prompt).json(), expected)
        self.assertEqual(calls[0].url.path, '/segment')
        self.assertFalse(app.state.lock.locked())


if __name__ == "__main__":
    unittest.main()

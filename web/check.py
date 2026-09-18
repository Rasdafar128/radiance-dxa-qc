"""Проверка границы веб-шлюза без GPU: python -m web.check."""

import io
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import httpx

from .app import app


CSV = ("path_to_study,study_uid,image_uid,anatomical_region,quality_class,quality_prob,"
       "violation_type,processing_status,time_of_processing\n"
       '"<script>.dcm",1,2,region,0,0.1,,Success,0.3\n')


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


if __name__ == "__main__":
    unittest.main()

"""Границы экспериментальной сегментации: python -m src.utils.check_segmentation."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np

from ..solution.segmentation import SegmentRequest, png_data


class SegmentationCheck(unittest.TestCase):
    def setUp(self):
        self.pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
        self.prompt = dict(image=png_data(self.pixels), box=[0, 0, 1, 1])

    def test_pixels_and_box_validation(self):
        result = SegmentRequest(**self.prompt).pixels()
        np.testing.assert_array_equal(result[:, :, 0], self.pixels)
        for box in ([0, 0, 0, 1], [1, 0, 0, 1], [-1, 0, 1, 1], [0, float('nan'), 1, 1]):
            with self.assertRaises(ValueError):
                SegmentRequest(**dict(self.prompt, box=box))
        with self.assertRaises(ValueError):
            SegmentRequest(**dict(self.prompt, box=[0, 0, .01, 1])).pixels()

    def test_malformed_and_oversize_images(self):
        for image in ('https://example.com/image.png', 'data:image/png;base64,invalid!',
                      png_data(np.zeros((2, 2), dtype=np.uint8)),
                      png_data(np.zeros((1025, 2), dtype=np.uint8))):
            with self.assertRaises((ValueError, OSError)):
                SegmentRequest(**dict(self.prompt, image=image)).pixels()

    def test_api_is_optional_lazy_and_serialized(self):
        from ..solution import api
        metadata = dict(name='Radiance', backbone='blend', version='1.0', recipe='test')
        classifier = SimpleNamespace(metadata=metadata, device='cpu', members=[])
        before = deepcopy(metadata)
        expected = dict(model='Radiance Anatomy', mask=self.prompt['image'], overlay=self.prompt['image'])
        with patch.object(api.Model, 'load', return_value=classifier), \
                patch.object(api, 'WEIGHTS') as weights, patch.object(api, 'Segmenter') as factory:
            factory.return_value.predict.side_effect = lambda pixels, box: (
                self.assertTrue(api.app.state.lock.locked()) or expected)
            with TestClient(api.app) as client:
                weights.is_file.return_value = False
                self.assertFalse(client.get('/health').json()['segmentation'])
                self.assertEqual(client.post('/segment', json=self.prompt).status_code, 503)
                factory.assert_not_called()
                weights.is_file.return_value = True
                for _ in range(2):
                    self.assertEqual(client.post('/segment', json=self.prompt).json(), expected)
                factory.assert_called_once_with(device='cpu')
                self.assertFalse(api.app.state.lock.locked())
                self.assertEqual(client.post('/segment', content=b'bad').status_code, 415)
                self.assertEqual(client.post('/segment', json={}).status_code, 400)
                with patch.object(api, 'MAX_REQUEST', 8):
                    self.assertEqual(client.post('/segment', json=self.prompt).status_code, 413)
                factory.return_value.predict.side_effect = RuntimeError('private details')
                response = client.post('/segment', json=self.prompt)
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('private details', response.text)
                self.assertFalse(api.app.state.lock.locked())
        self.assertEqual(metadata, before)


if __name__ == '__main__':
    unittest.main()

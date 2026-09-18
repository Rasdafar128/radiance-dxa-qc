"""SAM Large по рамке пользователя; отдельная маска, без изменения Radiance."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

MAX_REQUEST = 4 * 1024**2
WEIGHTS = Path(__file__).resolve().parents[2] / 'models/anatomy/sam-vit-large.pt'
SHA256 = '3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622'


class SegmentRequest(BaseModel):
    image: str = Field(max_length=MAX_REQUEST)
    box: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode='after')
    def valid_box(self):
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Choose a non-empty box inside the image')
        return self

    def pixels(self):
        prefix = 'data:image/png;base64,'
        if not self.image.startswith(prefix):
            raise ValueError('Expected a PNG preview')
        data = base64.b64decode(self.image[len(prefix):], validate=True)
        try:
            image = Image.open(BytesIO(data))
        except Image.DecompressionBombError:
            raise ValueError('Preview exceeds pixel limit') from None
        with image:
            if image.format != 'PNG' or min(image.size) < 2 or max(image.size) > 1024:
                raise ValueError('Preview must be a PNG up to 1024 pixels per side')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Only a single frame is supported')
            pixels = np.array(image.convert('RGB'))
        height, width = pixels.shape[:2]
        if (self.box[2] - self.box[0]) * width < 1 or (self.box[3] - self.box[1]) * height < 1:
            raise ValueError('Selection must cover at least one pixel per side')
        if np.ptp(pixels) == 0:
            raise ValueError('Cannot segment a constant image')
        return pixels


def png_data(pixels):
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


class Segmenter:
    def __init__(self, weights=WEIGHTS, device='cpu'):
        import hashlib
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        with Path(weights).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != SHA256:
                raise ValueError('SAM Large weights checksum mismatch')
        self.model = sam_model_registry['vit_l']()
        # mmap + assign не держат дополнительную копию checkpoint в RAM.
        self.model.load_state_dict(
            torch.load(weights, map_location='cpu', weights_only=True, mmap=True),
            strict=True, assign=True)
        self.model.to(device).eval()
        self.predictor = SamPredictor(self.model)

    def predict(self, pixels, box):
        import torch
        height, width = pixels.shape[:2]
        prompt = np.asarray(box) * [width, height, width, height]
        with torch.inference_mode():
            try:
                self.predictor.set_image(pixels)
                masks, _, _ = self.predictor.predict(box=prompt, multimask_output=False)
            finally:
                self.predictor.reset_image()
        mask = masks[0]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[mask] = (56, 189, 248, 110)
        return {'mask': png_data(mask.astype(np.uint8) * 255), 'overlay': png_data(overlay),
                'width': width, 'height': height, 'empty': not bool(mask.any()),
                'model': 'Radiance Anatomy', 'backbone': 'sam-vit-large', 'prompt': 'user_box'}

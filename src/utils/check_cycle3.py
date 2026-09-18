"""Regression and geometry checks for the final research cycle."""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image, ImageOps

from .. import config as C
from ..solution.dicom import pixels, prepare
from ..solution.model import preprocessing
from .train import inner_folds, tune, write_json
from .finetune import loss_terms


def run():
    logits = torch.tensor([[.1, -.2], [.4, .5], [-.3, .7]], requires_grad=True)
    labels = torch.tensor([[1., float('nan')], [0., 1.], [1., 0.]])
    weights = torch.tensor([.5, 1.])
    sums, counts = loss_terms(logits, labels, weights)
    (sums / counts).mean().backward()
    expected_gradient = logits.grad.clone()
    logits.grad.zero_()
    for index in range(3):
        sums, _ = loss_terms(logits[index:index+1], labels[index:index+1], weights)
        (sums / counts).mean().backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    assert logits.grad[0, 1] == 0 and torch.isfinite(logits.grad).all()
    root = C.ARTIFACTS / 'e2-dinov3-large'
    df = pd.read_csv(root / 'manifest.csv')
    with np.load(root / 'features.npz') as cache:
        x = cache['features']
    for fold in range(3):
        subset = df[df.fold != fold].reset_index(drop=True)
        split = pd.read_csv(root / f'inner_{fold}.csv')
        selected, inner, _, repeated = tune(subset, x[df.fold != fold], split)
        expected = json.loads((root / f'fold_{fold}/selection.json').read_text())['parameters']
        for values in expected.values():
            values.setdefault('solver', 'liblinear')
        assert selected == expected
        old = pd.read_csv(root / f'fold_{fold}/inner_predictions.csv').drop(columns='image_id')
        np.testing.assert_allclose(inner, old, atol=1e-12, rtol=1e-12)
        assert len(repeated) == len(subset)
        for seed in (42, 43, 44):
            assignments = inner_folds(subset, seed=seed)
            assert set(assignments.study) == set(subset.study)
            assert not set(assignments.study) & set(df[df.fold == fold].study)
    ds = pydicom.dcmread(C.DATA / df.path_to_study.iloc[0])
    recipe = preprocessing('dinov3-large')
    image = ImageOps.pad(Image.fromarray(pixels(ds)).convert('RGB'), (448, 448),
                        method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    expected = (np.asarray(image, dtype=np.float32)/255 - np.array(recipe['mean'], dtype=np.float32))/np.array(recipe['std'], dtype=np.float32)
    np.testing.assert_array_equal(prepare(ds, recipe), expected.transpose(2, 0, 1))
    synthetic = deepcopy(ds)
    synthetic.Rows, synthetic.Columns = 100, 100
    synthetic.BitsAllocated = synthetic.BitsStored = 16
    synthetic.HighBit, synthetic.PixelRepresentation = 15, 0
    synthetic.PhotometricInterpretation = 'MONOCHROME2'
    a = np.zeros((100, 100), dtype=np.uint16)
    a[10:90, 10:90] = 65535
    synthetic.PixelData = a.tobytes()
    plain = dict(recipe, mean=[0]*3, std=[1]*3)
    tensor = prepare(synthetic, dict(plain, aspect='physical'))
    yy, xx = np.where(tensor[0] > .5)
    ratio = (yy.max()-yy.min()+1)/(xx.max()-xx.min()+1)
    assert abs(ratio - C.PIXEL_MM_Y/C.PIXEL_MM_X) < .03
    assert tensor.shape == (3, 448, 448) and np.isfinite(tensor).all()
    result = dict(previous_tuning_reproduced=True, pixel_preprocessing_unchanged=True,
                  physical_square_observed_ratio=float(ratio), grouped_repeats_checked=True,
                  masked_loss_and_accumulation=True)
    output = C.ARTIFACTS / 'research-cycle-3'
    output.mkdir(exist_ok=True)
    write_json(output / 'implementation_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()

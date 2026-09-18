"""Reassemble E4 v1 inner scores by named heads; preserve all trained checkpoints.

python -m src.utils.repair_finetune artifacts/e4-dinov3 --output artifacts/e4-dinov3-v2
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..solution.model import HEADS, digest, probability
from .evaluate import evaluate
from .train import predictions, write_json
from .train_blend import cached_model, thresholds


def run(parent, output):
    assert (parent/'complete.json').exists() and not output.exists()
    original = json.loads((parent/'recipe.json').read_text())
    assert original['id'] == 'dinov3-last2-bf16-v1'
    for relative, checksum in original['code_sha256'].items():
        assert digest(parent/'source'/relative) == checksum
    def copy(source, destination):
        return os.link(source, destination) if str(source).endswith('.pt') else shutil.copy2(source, destination)
    shutil.copytree(parent, output, copy_function=copy, ignore=shutil.ignore_patterns('last.pt', '*.tmp', 'invalid.json'))
    recipe = dict(original, id='dinov3-last2-bf16-v2', correction_parent=str(parent),
                  correction_parent_recipe_sha256=digest(parent/'recipe.json'),
                  correction='Reassemble inner predictions by HEADS names; no training or outer-label selection')
    source = Path(__file__).resolve()
    relative = str(source.relative_to(C.ROOT))
    recipe['code_sha256'][relative] = digest(source)
    shutil.copy2(source, output/'source'/relative)
    write_json(output/'recipe.json', recipe)
    df = pd.read_csv(output/'manifest.csv')
    x = np.load(output/'features.npz', allow_pickle=False)['features']
    for fold in [0, 1, 2, 'final']:
        train = df if fold == 'final' else df[df.fold != fold]
        destination = output/('final' if fold == 'final' else f'fold_{fold}')
        selection = json.loads((destination/'selection.json').read_text())
        frames = []
        for inner_fold in (0, 1):
            job = output/'jobs'/f'{fold}_lr{selection["chosen_lr"]}_inner{inner_fold}'
            part = json.loads((job/'partition.json').read_text())
            scores = pd.read_csv(job/'predictions.csv')
            assert scores.image_id.tolist() == part['valid_ids']
            heads = json.loads((job/'heads.json').read_text())
            features = np.load(job/'validation_features.npy')
            for key in HEADS:
                np.testing.assert_allclose(scores[key], probability(features, heads[key]), atol=1e-12, rtol=1e-12)
            frames.append(scores.set_index('image_id')[HEADS])
        inner = pd.concat(frames)
        assert inner.index.is_unique and set(inner.index) == set(train.image_id)
        inner = inner.reindex(train.image_id)
        selected = thresholds(train, inner)
        selection['parameters'] = selected
        write_json(destination/'selection.json', selection)
        inner.to_csv(destination/'inner_predictions.csv', index_label='image_id')
        model = cached_model(destination)
        model.metadata.update(training=recipe, recipe=recipe['id'])
        for key, value in selected.items():
            model.heads[key]['threshold'] = value['threshold']
        write_json(destination/'model.json', dict(model.metadata, heads=model.heads))
        assert digest(destination/'encoder.pt') == model.metadata['encoder_sha256']
        if fold == 'final':
            inner.to_csv(output/'final_inner_predictions.csv', index_label='image_id')
        else:
            valid = df.fold == fold
            predictions(model, df[valid], x[valid]).to_csv(destination/'oof.csv', index=False)
            write_json(destination/'complete.json', dict(encoder_sha256=model.metadata['encoder_sha256'],
                       oof_sha256=digest(destination/'oof.csv')))
    oof = pd.concat([pd.read_csv(output/f'fold_{k}/oof.csv') for k in range(3)], ignore_index=True)
    oof.to_csv(output/'oof.csv', index=False)
    report = evaluate(oof, 2000)
    write_json(output/'metrics.json', report)
    write_json(output/'complete.json', dict(recipe_sha256=digest(output/'recipe.json'), metrics_sha256=digest(output/'metrics.json')))
    write_json(parent/'invalid.json', dict(reason='Inner-head column permutation; metrics invalid', corrected_run=str(output)))
    print(json.dumps(dict(quality=report['metrics']['quality_all'], macro_f1=report['macro_f1'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output)

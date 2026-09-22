"""Reproduce the merged Radiance recipe on the original grouped outer/inner folds.

python -m src.utils.train_geometry --parent artifacts/e5-blend --output artifacts/unified
Run again with e5-blend-split137 for the secondary split. Requires parent artifacts.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pydicom

from .. import config as C
from ..solution.dicom import pixels
from ..solution.geometry import COLUMNS, TASK_COLUMNS, features
from ..solution.model import GeometryBlend, digest, fit_head, probability, targets
from .evaluate import evaluate
from .train import predictions, write_json
from .train_blend import cached_model, thresholds


def fit_geometry(x, y):
    fill = np.array([np.mean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    values = np.where(np.isfinite(x), x, fill)
    head = fit_head(values, y, regularization=1., balanced=True, solver='liblinear')
    return dict(head, impute=fill.tolist())


def geometry_probability(x, head):
    return probability(np.where(np.isfinite(x), x, head['impute']), head)


def run(parent, output, resamples=2000):
    df = pd.read_csv(parent / 'manifest.csv')
    assert len(df) == 249 and df.study.nunique() == 100
    assert df.image_id.is_unique and df.pixel_sha256.is_unique
    assert df.groupby('study').fold.nunique().eq(1).all()
    with np.load(parent / 'features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['image_ids'], df.image_id)
        base_features = cache['features']
    geom = []
    for row in df.itertuples():
        path = C.DATA / row.path_to_study
        assert digest(path) == row.file_sha256
        geom.append(features(pixels(pydicom.dcmread(path)), row.anatomical_region == C.REGION_FEMUR))
    geom = np.asarray(geom)
    x = np.concatenate([base_features, geom], axis=1)
    recipe = dict(id='radiance-geometry-v1', parent=str(parent),
                  parent_recipe_sha256=digest(parent / 'recipe.json'),
                  parent_features_sha256=digest(parent / 'features.npz'),
                  manifest_sha256=digest(parent / 'manifest.csv'),
                  geometry_columns=COLUMNS, geometry_tasks=TASK_COLUMNS,
                  geometry_head=dict(C=1., balanced=True, solver='liblinear', imputation='train_mean'),
                  quality='half direct probability plus half noisy-OR of criteria',
                  selection='Development OOF recipe selection on two splits; thresholds on inner train only',
                  independent_test=False,
                  versions={p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','scipy','opencv-python-headless']},
                  code_sha256={p:digest(C.ROOT / p) for p in ['src/solution/geometry.py','src/solution/model.py','src/utils/train_geometry.py']})
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'recipe.json').exists() and json.loads((output / 'recipe.json').read_text()) != recipe:
        raise ValueError('Run changed; use a new output directory')
    write_json(output / 'recipe.json', recipe)
    shutil.copyfile(parent / 'manifest.csv', output / 'manifest.csv')
    for p in recipe['code_sha256']:
        dst = output / 'source' / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / p, dst)
    np.savez_compressed(output / 'features.npz', features=x, image_ids=df.image_id.to_numpy(str))
    for fold in [0, 1, 2, 'final']:
        train = df if fold == 'final' else df[df.fold != fold]
        folder = 'final' if fold == 'final' else f'fold_{fold}'
        inner_split = pd.read_csv(parent / f'inner_{fold}.csv')
        assert inner_split.study.is_unique and set(inner_split.study) == set(train.study)
        assignments = train.study.map(inner_split.set_index('study').fold).to_numpy()
        inner = pd.read_csv(parent / folder / 'inner_predictions.csv').set_index('image_id').reindex(train.image_id)
        heads = {}
        for key, columns in TASK_COLUMNS.items():
            y = targets(train)[key]
            known = np.isfinite(y)
            values = geom[train.index][:, columns]
            for k in [0, 1]:
                a, b = known & (assignments != k), known & (assignments == k)
                assert not set(train.study.to_numpy()[a]) & set(train.study.to_numpy()[b])
                head = fit_geometry(values[a], y[a])
                inner.loc[train.image_id.to_numpy()[b], key] = geometry_probability(values[b], head)
            heads[key] = fit_geometry(values[known], y[known])
        for quality, types in [('spine_quality', C.TARGETS[:3]), ('hip_quality', C.TARGETS[3:])]:
            inner[quality] = .5 * inner[quality] + .5 * (1 - np.prod(1 - inner[types], axis=1))
        selected = thresholds(train, inner)
        members = [cached_model(parent / folder / f'member_{i}') for i in range(2)]
        model = GeometryBlend(members, selected, heads)
        model.metadata.update(name='Radiance', version='1.0', training=recipe,
                              training_partition='all_labeled' if fold == 'final' else f'outer_train_{fold}')
        dest = output / folder
        dest.mkdir(exist_ok=True)
        for i in range(2):
            source, target = parent / folder / f'member_{i}', dest / f'member_{i}'
            target.mkdir(exist_ok=True)
            shutil.copyfile(source / 'model.json', target / 'model.json')
            if not (target / 'encoder.pt').exists():
                os.link(source / 'encoder.pt', target / 'encoder.pt')
        write_json(dest / 'model.json', dict(model.metadata, heads=selected))
        inner.to_csv(dest / 'inner_predictions.csv', index_label='image_id')
        shutil.copyfile(parent / f'inner_{fold}.csv', output / f'inner_{fold}.csv')
        if fold != 'final':
            valid = df[df.fold == fold]
            predictions(model, valid, x[valid.index]).to_csv(dest / 'oof.csv', index=False)
        print(f'{output.name}: {folder} complete', flush=True)
    oof = pd.concat([pd.read_csv(output / f'fold_{k}/oof.csv') for k in range(3)], ignore_index=True)
    assert len(oof) == len(df) and oof.image_id.is_unique
    oof.to_csv(output / 'oof.csv', index=False)
    report = evaluate(oof, resamples)
    write_json(output / 'metrics.json', report)
    print(json.dumps(dict(f1=report['metrics']['quality_all']['f1'],auc=report['metrics']['quality_all']['roc_auc'],macro=report['macro_f1'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=C.ARTIFACTS / 'e5-blend')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    args = parser.parse_args()
    run(args.parent, args.output, args.resamples)

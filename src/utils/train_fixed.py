"""Matched frozen control for E4: fixed warm heads, identical nested thresholds."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .evaluate import evaluate
from .train import inner_folds, predictions, write_json
from .train_blend import cached_model, thresholds
from ..solution.model import digest


def run(output):
    parent = C.ARTIFACTS / 'e2-dinov3-large'
    df = pd.read_csv(parent/'manifest.csv')
    with np.load(parent/'features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['image_ids'], df.image_id)
        x = cache['features']
    output.mkdir(parents=True, exist_ok=True)
    code = [C.ROOT/'src'/p for p in ('config.py', 'solution/model.py', 'solution/dicom.py',
            'utils/train_fixed.py', 'utils/train.py', 'utils/train_blend.py', 'utils/evaluate.py')]
    recipe = dict(id='dinov3-fixed-warm-control-v1', manifest_sha256=digest(parent/'manifest.csv'),
                  parent_recipe_sha256=digest(parent/'recipe.json'), features_sha256=digest(parent/'features.npz'),
                  warm_head=dict(C=.1, balanced=True, solver='liblinear'), quality_gate=True,
                  code_sha256={str(p.relative_to(C.ROOT)): digest(p) for p in code})
    if (output/'recipe.json').exists():
        assert json.loads((output/'recipe.json').read_text()) == recipe
    write_json(output/'recipe.json', recipe)
    for source in code:
        archive = output/'source'/source.relative_to(C.ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, archive)
    shutil.copyfile(parent/'manifest.csv', output/'manifest.csv')
    shutil.copyfile(parent/'features.npz', output/'features.npz')
    if not (output/'encoder.pt').exists():
        os.link(parent/'encoder.pt', output/'encoder.pt')
    model = cached_model(parent/'final')
    model.metadata.update(training=recipe, recipe=recipe['id'], quality_gate=True)
    for fold in [0, 1, 2, 'final']:
        train = df if fold == 'final' else df[df.fold != fold]
        split = inner_folds(train.reset_index(drop=True))
        split.to_csv(output/f'inner_{fold}.csv', index=False)
        assignments = train.study.map(split.set_index('study').fold)
        inner = pd.DataFrame(index=train.index)
        for k in (0, 1):
            a, b = train[assignments != k], train[assignments == k]
            assert not set(a.study) & set(b.study)
            model.fit(a, features=x[a.index], verbose=False)
            _, scores = model.classify(x[b.index])
            for key, values in scores.items():
                inner.loc[b.index, key] = values
        selected = thresholds(train, inner)
        model.fit(train, features=x[train.index], verbose=False)
        for key, value in selected.items():
            model.heads[key]['threshold'] = value['threshold']
        model.metadata['training_partition'] = 'all_labeled' if fold == 'final' else f'outer_train_{fold}'
        destination = output/('final' if fold == 'final' else f'fold_{fold}')
        model.save(destination, encoder_path=output/'encoder.pt')
        write_json(destination/'selection.json', dict(parameters=selected))
        inner.index = train.image_id
        inner.to_csv(destination/'inner_predictions.csv', index_label='image_id')
        if fold == 'final':
            inner.to_csv(output/'final_inner_predictions.csv', index_label='image_id')
        else:
            valid = df[df.fold == fold]
            predictions(model, valid, x[valid.index]).to_csv(destination/'oof.csv', index=False)
            write_json(destination/'complete.json', dict(oof_sha256=digest(destination/'oof.csv')))
    oof = pd.concat([pd.read_csv(output/f'fold_{k}/oof.csv') for k in range(3)], ignore_index=True)
    oof.to_csv(output/'oof.csv', index=False)
    report = evaluate(oof, 2000)
    write_json(output/'metrics.json', report)
    write_json(output/'complete.json', dict(recipe_sha256=digest(output/'recipe.json'), metrics_sha256=digest(output/'metrics.json')))
    print(json.dumps(dict(quality=report['metrics']['quality_all'], macro_f1=report['macro_f1'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)

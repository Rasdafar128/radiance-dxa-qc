"""Second exploratory pass: finer occlusion and conservative two-fill agreement."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import pydicom
import torch

from .. import config as C
from ..solution.dicom import pixels
from ..solution.model import Model
from .localization import RUN, cams, cells, correlation, normalized, render, replace, score


def run(args):
    torch.set_num_threads(2)
    root = args.output
    cases = json.loads((root / 'cases.json').read_text())
    models = [Model.load(RUN / f'fold_0/member_{i}', args.device) for i in range(2)]
    for model in models:
        model.encoder.requires_grad_(False)
    for case in cases:
        if case['case'] not in args.cases:
            continue
        started = time.monotonic()
        name = case['case']
        for i, model in enumerate(models):
            metadata = json.loads((RUN / f'fold_{case["fold"]}/member_{i}/model.json').read_text())
            assert metadata['encoder_sha256'] == model.metadata['encoder_sha256']
            model.heads = metadata['heads']
        a = pixels(pydicom.dcmread(C.DATA / case['path']))
        base = score(models, a, case['target'])
        assert abs(base - case['probability']) < 1e-4
        fills = {'blur': np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(max(a.shape)*.08))),
                 'mean': np.full_like(a, round(float(a.mean())))}
        if args.hybrid_only:
            verified = json.loads((root / f'{name}-refined.json').read_text())
            assert verified['target'] == case['target'] and abs(verified['probability'] - base) < 1e-4
            with np.load(root / f'{name}-maps.npz') as stored:
                gradient = stored['activation_grad']
            with np.load(root / f'{name}-refined.npz') as stored:
                agreement = stored['agreement']
            heat = agreement * gradient
            regions = cells(a, 12)
            values = np.array([heat[r].mean() for r in regions])
            positive = np.flatnonzero(values > 0)
            order = positive[np.argsort(values[positive])[-4:]]
            drops = {}
            for fill, replacement in fills.items():
                modified = replace(a, replacement, [regions[i] for i in order])
                drops[fill] = base - score(models, modified, case['target'])
                Image.fromarray(modified).save(root / f'{name}-hybrid-{fill}.png')
            boxes = [[int(x) for x in (r[1].start, r[0].start, r[1].stop, r[0].stop)]
                     for r in (regions[i] for i in order)]
            result = dict(case=name, target=case['target'], outcome=case['outcome'], probability=base,
                          selected_cells=len(order), boxes=boxes, drops=drops)
            (root / f'{name}-hybrid.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            render(a, {'gradient': gradient, 'agreement': agreement, 'hybrid': heat},
                   f'{name} gradient x verified occlusion; exploratory fusion', root / f'{name}-hybrid.jpg')
            print(name, 'hybrid DONE', flush=True)
            continue
        target_maps = {}
        for target in C.TARGETS:
            if target.split('_')[0] == case['target'].split('_')[0]:
                target_maps[target] = np.mean([normalized(cams(m, a, target)['activation_grad']) for m in models], axis=0)
        class_similarity = {target: correlation(target_maps[case['target']], heat)
                            for target, heat in target_maps.items() if target != case['target']}
        render(a, target_maps, f'{name} same image, different violation heads', root / f'{name}-targets.jpg')
        regions = cells(a, 12)
        maps, values = {}, {}
        for fill, replacement in fills.items():
            values[fill] = np.array([base - score(models, replace(a, replacement, [region]), case['target'])
                                    for region in regions])
            heat = np.zeros_like(a, dtype=np.float32)
            for region, delta in zip(regions, values[fill]):
                heat[region] = delta
            maps['fine_' + fill] = heat
            print(name, fill, '12x12 done', flush=True)
        consensus = np.minimum(np.maximum(values['blur'], 0), np.maximum(values['mean'], 0))
        maps['agreement'] = np.minimum(np.maximum(maps['fine_blur'], 0), np.maximum(maps['fine_mean'], 0))
        positive = np.flatnonzero(consensus > 0)
        order = positive[np.argsort(consensus[positive])[-4:]]
        random_sets = [np.random.default_rng(seed).choice(len(regions), len(order), replace=False)
                       for seed in (41, 42, 43)]
        drops = {}
        for fill, replacement in fills.items():
            modified = replace(a, replacement, [regions[i] for i in order])
            drops[fill] = {'top4_drop': base - score(models, modified, case['target']),
                           'random4_drops': [base - score(models, replace(a, replacement, [regions[i] for i in s]), case['target'])
                                             for s in random_sets]}
            Image.fromarray(modified).save(root / f'{name}-counterfactual-{fill}.png')
        bounds = [[int(x) for x in (r[1].start, r[0].start, r[1].stop, r[0].stop)]
                  for r in (regions[i] for i in order)]
        result = dict(case=name, target=case['target'], outcome=case['outcome'], probability=base,
                      grid=12, selected_cells=len(order), top4_boxes=bounds, deletion=drops,
                      baseline_spearman=correlation(values['blur'], values['mean']),
                      other_head_spearman=class_similarity,
                      max_agreement_delta=float(consensus.max()), seconds=time.monotonic()-started)
        (root / f'{name}-refined.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        np.savez_compressed(root / f'{name}-refined.npz', **maps)
        render(a, maps, f'{name} {case["target"]} {case["outcome"]} | 12x12 interventions, not masks', root / f'{name}-refined.jpg')
        print(name, 'refinement DONE', round(result['seconds'], 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=C.ARTIFACTS / 'localization')
    parser.add_argument('--cases', nargs='+', default=['case-08', 'case-09', 'case-16', 'case-17'])
    parser.add_argument('--hybrid-only', action='store_true', help='Reuse completed maps for the fusion experiment')
    run(parser.parse_args())

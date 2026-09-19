"""Research only: held-out violation attribution; never a segmentation mask.

python -m src.utils.localization --device mps --output artifacts/localization
"""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import pydicom
from scipy.stats import spearmanr
import torch

from .. import config as C
from ..solution.dicom import pixels, prepare
from ..solution.model import Model

RUN = C.ARTIFACTS / 'e5-blend'
GRID = 6


def select_cases():
    oof = pd.read_csv(RUN / 'oof.csv')
    cases, used = [], set()
    for target in C.TARGETS:
        for outcome, truth, prediction in [('TP', 1, 1), ('FP', 0, 1), ('FN', 1, 0), ('TN', 0, 0)]:
            rows = oof[(oof['true_' + target] == truth) & (oof['pred_' + target] == prediction)].sort_values('image_id')
            fresh = rows[~rows.study.isin(used)]
            if len(fresh):
                rows = fresh
            if rows.empty:
                continue
            row = rows.iloc[0]
            used.add(row.study)
            cases.append(dict(case=f'case-{len(cases)+1:02d}', target=target, outcome=outcome,
                              fold=int(row.fold), image_id=row.image_id, path=row.path_to_study,
                              study=row.study, probability=float(row['prob_' + target]),
                              label=truth, prediction=prediction))
    return cases


def tensor_from_pixels(a, model):
    recipe = model.metadata['preprocess']
    assert recipe['aspect'] == 'pixel_grid'
    im = ImageOps.pad(Image.fromarray(a).convert('RGB'), (recipe['size'],) * 2,
                      method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    x = np.asarray(im, dtype=np.float32) / 255
    x = (x - np.asarray(recipe['mean'], dtype=np.float32)) / np.asarray(recipe['std'], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(model.device)


def forward(model, x):
    output = model.encoder(x)
    return output.last_hidden_state[:, 0] if model.metadata['backbone'] == 'dinov3-large' else output


def head_probability(features, head):
    weights = features.new_tensor(head['coef'])
    return torch.sigmoid(features @ weights + head['intercept'])


def score(models, a, target):
    with torch.no_grad():
        return float(np.mean([head_probability(forward(m, tensor_from_pixels(a, m)), m.heads[target]).item()
                              for m in models]))


def restore_map(grid, a, size):
    height, width = a.shape
    fitted = ImageOps.contain(Image.new('L', (width, height)), (size, size))
    left, top = round((size - fitted.width) / 2), round((size - fitted.height) / 2)
    canvas = Image.fromarray(grid.astype(np.float32)).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(canvas.crop((left, top, left + fitted.width, top + fitted.height))
                      .resize((width, height), Image.Resampling.BILINEAR)).copy()


def normalized(a):
    a = np.maximum(a, 0)
    return a / max(float(a.max()), 1e-12)


def cams(model, a, target, randomize=False):
    captured = []
    dino = model.metadata['backbone'] == 'dinov3-large'
    layer = model.encoder.layer[-1].norm1 if dino else model.encoder.image_encoder.blocks[-1]

    def capture(module, inputs, output):
        activation = output if dino else output[0]
        activation.requires_grad_(True)
        captured.append(activation)

    handle = layer.register_forward_hook(capture)
    try:
        features = forward(model, tensor_from_pixels(a, model))
        head = dict(model.heads[target])
        if randomize:
            head['coef'] = np.random.default_rng(42).permutation(head['coef']).tolist()
        logit = features @ features.new_tensor(head['coef']) + head['intercept']
        activation = captured[0]
        gradient, = torch.autograd.grad(logit.sum(), activation)
        start = 1 + model.encoder.config.num_register_tokens if dino else 0
        act = activation[0, start:].detach().cpu().numpy()
        grad = gradient[0, start:].detach().cpu().numpy()
        side = int(len(act) ** .5)
        assert side * side == len(act)
        maps = {'gradcam': (act * grad.mean(axis=0)).sum(axis=-1),
                'activation_grad': (act * grad).sum(axis=-1)}
        return {key: restore_map(np.maximum(value, 0).reshape(side, side), a, model.metadata['preprocess']['size'])
                for key, value in maps.items()}
    finally:
        handle.remove()


def cells(a, grid=GRID):
    ys = np.linspace(0, a.shape[0], grid + 1).astype(int)
    xs = np.linspace(0, a.shape[1], grid + 1).astype(int)
    return [(slice(ys[y], ys[y+1]), slice(xs[x], xs[x+1])) for y in range(grid) for x in range(grid)]


def replace(a, replacement, regions):
    result = a.copy()
    for region in regions:
        result[region] = replacement[region]
    return result


def correlation(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def top_iou(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    n = max(1, round(a.size * .2))
    x, y = np.argsort(a.ravel())[-n:], np.argsort(b.ravel())[-n:]
    return float(len(np.intersect1d(x, y)) / len(np.union1d(x, y)))


def check_geometry():
    # Odd padding must use Pillow's round, not integer division.
    for height, width in [(317, 300), (291, 280), (206, 280), (101, 200), (200, 101)]:
        for size in (448, 512):
            a = np.zeros((height, width), dtype=np.uint8)
            padded = np.asarray(ImageOps.pad(Image.fromarray(a + 255), (size, size))) / 255
            assert np.all(restore_map(padded, a, size) == 1)
            coverage = np.zeros_like(a)
            for region in cells(a):
                coverage[region] += 1
            assert np.all(coverage == 1)
            assert np.all(replace(a, a + 1, cells(a)) == 1) and not a.any()
    assert top_iou(np.zeros((4, 4)), np.zeros((4, 4))) is None


def render(a, maps, title, path):
    # Sequential opacity, fixed hue; panels explicitly distinguish independent scales.
    names = ['original', *maps]
    canvas = Image.new('RGB', (260 * len(names), 354), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    draw.text((8, 5), title, fill='black', font=font)
    for index, name in enumerate(names):
        rgb = np.repeat(a[..., None], 3, axis=-1).astype(float)
        if name != 'original':
            alpha = normalized(maps[name])[..., None] * .65
            rgb = rgb * (1 - alpha) + np.array([255, 60, 0]) * alpha
        im = ImageOps.contain(Image.fromarray(np.uint8(np.clip(rgb, 0, 255))), (250, 290))
        x = index * 260
        canvas.paste(im, (x + (260 - im.width) // 2, 52))
        draw.text((x + 5, 30), name, fill='black', font=font)
    canvas.save(path)


def summarize(reports):
    def median(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    result = dict(images=len(reports), studies=len({r['case']['study'] for r in reports}),
                  protocol='stratified development OOF diagnostic panel; no reference masks',
                  max_oof_probability_error=max(abs(r['probability'] - r['case']['probability']) for r in reports),
                  median_case_seconds=median([r['seconds'] for r in reports]), cohorts={})
    for cohort, rows in [('all', reports), ('predicted_positive', [r for r in reports if r['case']['prediction']])]:
        entry = dict(images=len(rows), methods={})
        for method in ('gradcam', 'activation_grad', 'occlusion_blur', 'occlusion_mean'):
            measures = {}
            for fill in ('blur', 'mean'):
                valid = [r for r in rows if r['deletion'][fill][method] is not None]
                top = [r['deletion'][fill][method]['top'] for r in valid]
                bottom = [r['deletion'][fill][method]['bottom'] for r in valid]
                random = [np.mean(r['deletion'][fill]['random']) for r in valid]
                measures[fill] = dict(n=len(valid), median_top_drop_pp=100*median(top) if valid else None,
                                      median_random_drop_pp=100*median(random) if valid else None,
                                      top_reduces_score=int(sum(t > 0 for t in top)),
                                      positive_drop_beats_random=int(sum(t > 0 and t > q for t, q in zip(top, random))),
                                      top_beats_random=int(sum(t > q for t, q in zip(top, random))),
                                      top_beats_bottom=int(sum(t > b for t, b in zip(top, bottom))))
            if method in ('gradcam', 'activation_grad'):
                measures['head_shuffle_spearman_median'] = median([r['checks'][method]['head_shuffle_spearman'] for r in rows])
                measures['shift4_top20_iou_median'] = median([r['checks'][method]['shift4_top20_iou'] for r in rows])
            entry['methods'][method] = measures
        entry['occlusion_baseline_spearman_median'] = median([r['checks']['occlusion']['baseline_spearman'] for r in rows])
        entry['occlusion_baseline_top20_iou_median'] = median([r['checks']['occlusion']['baseline_top20_iou'] for r in rows])
        result['cohorts'][cohort] = entry
    return result


def run(args):
    check_geometry()
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    selected = select_cases()
    (out / 'cases.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    print('Loading frozen encoders; heads selected per outer fold', flush=True)
    models = [Model.load(RUN / f'fold_0/member_{i}', args.device) for i in range(2)]
    for model in models:
        model.encoder.requires_grad_(False)
    manifest = pd.read_csv(RUN / 'manifest.csv')
    reports = []
    for case in selected:
        if args.cases and case['case'] not in args.cases:
            continue
        name = case['case']
        if (out / f'{name}.json').exists() and not args.overwrite:
            reports.append(json.loads((out / f'{name}.json').read_text()))
            continue
        started = time.monotonic()
        for i, model in enumerate(models):
            metadata = json.loads((RUN / f'fold_{case["fold"]}/member_{i}/model.json').read_text())
            assert metadata['encoder_sha256'] == model.metadata['encoder_sha256']
            model.heads = metadata['heads']
        fold_metadata = json.loads((RUN / f'fold_{case["fold"]}/model.json').read_text())
        assert fold_metadata['training_partition'] == f'outer_train_{case["fold"]}'
        assert not manifest[(manifest.fold != case['fold']) & (manifest.study == case['study'])].shape[0]
        ds = pydicom.dcmread(C.DATA / case['path'])
        a = pixels(ds)
        for model in models:
            np.testing.assert_array_equal(tensor_from_pixels(a, model).cpu().numpy()[0], prepare(ds, model.metadata['preprocess']))
        base = score(models, a, case['target'])
        assert abs(base - case['probability']) < 1e-4, (name, base, case['probability'])
        maps, branch_maps = {}, []
        for model in models:
            branch_maps.append(cams(model, a, case['target']))
        for method in branch_maps[0]:
            maps[method] = np.mean([normalized(b[method]) for b in branch_maps], axis=0)
        maps['dino_activation_grad'] = normalized(branch_maps[0]['activation_grad'])
        maps['med_activation_grad'] = normalized(branch_maps[1]['activation_grad'])
        randomized = [cams(m, a, case['target'], randomize=True) for m in models]
        random_maps = {key: np.mean([normalized(b[key]) for b in randomized], axis=0)
                       for key in ('gradcam', 'activation_grad')}
        shifted = np.pad(a, ((0, 0), (4, 0)), mode='constant')[:, :a.shape[1]]
        shifted_branches = [cams(m, shifted, case['target']) for m in models]
        shifted_maps = {key: np.mean([normalized(b[key]) for b in shifted_branches], axis=0)
                        for key in ('gradcam', 'activation_grad')}
        checks = {key: dict(head_shuffle_spearman=correlation(maps[key], random_maps[key]),
                           shift4_top20_iou=top_iou(maps[key][:, :-4], shifted_maps[key][:, 4:]))
                  for key in random_maps}
        replacements = {'blur': np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=max(a.shape)*.08))),
                        'mean': np.full_like(a, round(float(a.mean())))}
        regions = cells(a)
        for baseline, replacement in replacements.items():
            deltas = np.array([base - score(models, replace(a, replacement, [region]), case['target'])
                               for region in regions])
            heat = np.zeros_like(a, dtype=np.float32)
            for region, delta in zip(regions, deltas):
                heat[region] = delta
            maps['occlusion_' + baseline] = heat
            print(name, baseline, 'occlusion complete', flush=True)
        checks['occlusion'] = dict(baseline_spearman=correlation(maps['occlusion_blur'], maps['occlusion_mean']),
                                   baseline_top20_iou=top_iou(maps['occlusion_blur'], maps['occlusion_mean']))
        deletion = {}
        rng = np.random.default_rng(20260919)
        random_sets = [rng.choice(len(regions), 7, replace=False) for _ in range(3)]
        for baseline, replacement in replacements.items():
            random_drops = [base - score(models, replace(a, replacement, [regions[i] for i in indices]), case['target'])
                            for indices in random_sets]
            deletion[baseline] = {'random': random_drops}
            for method in ('gradcam', 'activation_grad', 'occlusion_blur', 'occlusion_mean'):
                values = np.array([maps[method][region].mean() for region in regions])
                if np.ptp(maps[method]) < 1e-12:
                    deletion[baseline][method] = None
                    continue
                order = np.argsort(values)
                deletion[baseline][method] = {
                    label: base - score(models, replace(a, replacement, [regions[i] for i in indices]), case['target'])
                    for label, indices in [('top', order[-7:]), ('bottom', order[:7])]}
        report = dict(case=case, probability=base, checks=checks, deletion=deletion,
                      branch_probability=[score([m], a, case['target']) for m in models],
                      map_ranges={k: [float(v.min()), float(v.max())] for k, v in maps.items()},
                      seconds=round(time.monotonic()-started, 2))
        np.savez_compressed(out / f'{name}-maps.npz', **maps, **{'random_'+k:v for k,v in random_maps.items()})
        render(a, {k:maps[k] for k in ('gradcam','activation_grad','occlusion_blur','occlusion_mean')},
               f'{name} {case["target"]} {case["outcome"]} p={base:.3f} | independent positive scales; not lesion masks', out / f'{name}.jpg')
        render(a, {k:maps[k] for k in ('dino_activation_grad','med_activation_grad')},
               f'{name} encoder comparison', out / f'{name}-branches.jpg')
        (out / f'{name}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        reports.append(report)
        print(name, 'DONE', report['seconds'], 'seconds', flush=True)
        gc.collect()
    (out / 'results.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False))
    (out / 'summary.json').write_text(json.dumps(summarize(reports), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=C.ARTIFACTS / 'localization')
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())

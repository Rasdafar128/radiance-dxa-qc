"""Nested DINOv3-L adaptation: last two blocks, train-only initialization and selection.

python -m src.utils.finetune --output artifacts/e4-dinov3
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
from io import StringIO
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.nn import functional as F

from .. import config as C
from ..solution.dicom import prepare
from ..solution.model import HEADS, Model, digest, targets
from .evaluate import evaluate
from .train import inner_folds, manifest, predictions, write_json
from .train_blend import thresholds


def tensor_targets(df):
    values = targets(df)
    return torch.tensor(np.stack([values[key] for key in HEADS], axis=1), dtype=torch.float32)


def loss_terms(logits, labels, positive_weight):
    known = labels.isfinite()
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.nan_to_num(),
                                                pos_weight=positive_weight, reduction='none')
    return (losses * known).sum(0), known.sum(0)


def save_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def head_module(model):
    layer = torch.nn.Linear(1024, len(HEADS)).to(model.device)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([model.heads[k]['coef'] for k in HEADS], device=model.device))
        layer.bias.copy_(torch.tensor([model.heads[k]['intercept'] for k in HEADS], device=model.device))
    return layer


def export_heads(model, layer):
    for index, key in enumerate(HEADS):
        model.heads[key].update(coef=layer.weight[index].detach().cpu().tolist(),
                                intercept=float(layer.bias[index].detach()))


def delta(model, head):
    return dict(encoder={k: p.detach().cpu().clone() for k, p in model.encoder.named_parameters() if p.requires_grad},
                head={k: p.detach().cpu().clone() for k, p in head.state_dict().items()})


def restore(model, head, state):
    expected = {k for k, p in model.encoder.named_parameters() if p.requires_grad}
    if set(state['encoder']) != expected:
        raise ValueError('Checkpoint does not contain exactly the adapted blocks')
    model.encoder.load_state_dict(state['encoder'], strict=False)
    head.load_state_dict(state['head'], strict=True)


@torch.no_grad()
def validation_loss(model, head, images, labels, indices, weight):
    model.encoder.eval()
    sums = torch.zeros(len(HEADS), device=model.device)
    counts = torch.zeros_like(sums)
    for index in np.array_split(indices, max(1, (len(indices)+1)//2)):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = model.encoder(images[index].to(model.device)).last_hidden_state[:, 0]
            values, known = loss_terms(head(features), labels[index].to(model.device), weight)
        sums += values
        counts += known
    assert (counts > 0).all()
    value = float((sums / counts).mean())
    assert np.isfinite(value)
    return value


def fit_partition(base, df, features, images, train, valid, lr, destination, epochs=None):
    """No outer-validation labels are passed to a refit (valid is empty)."""
    destination.mkdir(parents=True, exist_ok=True)
    assert len(set(train) & set(valid)) == 0
    assert not set(df.iloc[train].study) & set(df.iloc[valid].study)
    torch.manual_seed(42)
    model = Model('cuda', backbone='dinov3-large', encoder_config=base['encoder_config'])
    model.encoder.load_state_dict(torch.load(C.ARTIFACTS / 'e2-dinov3-large/encoder.pt', map_location='cpu', weights_only=True))
    model.encoder.requires_grad_(False)
    for block in model.encoder.layer[-2:]:
        block.requires_grad_(True)
    model.fit(df.iloc[train], features=features[train], verbose=False)
    head = head_module(model)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=lr),
        dict(params=head.parameters(), lr=lr*10)], weight_decay=.01)
    labels = tensor_targets(df)
    known = labels[train].isfinite().sum(0).to('cuda')
    positive = (labels[train] == 1).sum(0).to('cuda')
    negative = (labels[train] == 0).sum(0).to('cuda')
    assert ((positive > 0) & (negative > 0)).all()
    weight = negative / positive
    partition = dict(train_ids=df.iloc[train].image_id.tolist(), valid_ids=df.iloc[valid].image_id.tolist(),
                     train_studies=sorted(df.iloc[train].study.unique().tolist()),
                     valid_studies=sorted(df.iloc[valid].study.unique().tolist()), lr=lr, refit_epochs=epochs,
                     positive_weight=weight.cpu().tolist(), known_counts=known.cpu().tolist(),
                     code_sha256=digest(Path(__file__)), source_encoder_sha256=base['encoder_sha256'])
    if (destination / 'partition.json').exists():
        assert json.loads((destination / 'partition.json').read_text()) == partition
    write_json(destination / 'partition.json', partition)
    partition_sha256 = digest(destination / 'partition.json')
    best_path, last_path = destination / 'best.pt', destination / 'last.pt'
    complete = destination / 'complete.json'
    if complete.exists():
        summary = json.loads(complete.read_text())
        assert summary['partition_sha256'] == partition_sha256
        assert digest(best_path) == summary['best_sha256']
        restore(model, head, torch.load(best_path, map_location='cpu', weights_only=True))
        export_heads(model, head)
        model.encoder.eval()
        return model, summary
    history, start_epoch, best_epoch, stale = [], 1, 0, 0
    best_loss = validation_loss(model, head, images, labels, valid, weight) if len(valid) else None
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=True)
        assert checkpoint['partition_sha256'] == partition_sha256
        restore(model, head, checkpoint['state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        history, start_epoch = checkpoint['history'], checkpoint['epoch'] + 1
        best_epoch, best_loss, stale = checkpoint['best_epoch'], checkpoint['best_loss'], checkpoint['stale']
    else:
        save_torch(best_path, delta(model, head))
        history.append(dict(epoch=0, validation_loss=best_loss))
    started = perf_counter()
    for epoch in range(start_epoch, (20 if epochs is None else epochs) + 1):
        if len(valid) and stale >= 4:
            break
        model.encoder.train()
        order = np.random.default_rng(42 + epoch).permutation(train)
        total = np.zeros(len(HEADS))
        for offset in range(0, len(order), 16):
            group = order[offset:offset+16]
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, len(group), 2):
                index = group[micro:micro+2]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    encoded = model.encoder(images[index].to('cuda')).last_hidden_state[:, 0]
                    sums, _ = loss_terms(head(encoded), labels[index].to('cuda'), weight)
                    # Unbiased per-head average with fixed train-only denominators.
                    loss = (sums * len(train) / known).mean() / len(group)
                assert torch.isfinite(loss)
                loss.backward()
                total += sums.detach().cpu().numpy()
            assert all(p.grad is None for p in model.encoder.layer[0].parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for group_params in optimizer.param_groups for p in group_params['params'])
            optimizer.step()
        val = validation_loss(model, head, images, labels, valid, weight) if len(valid) else None
        history.append(dict(epoch=epoch, train_loss=float((total / known.cpu().numpy()).mean()),
                            validation_loss=val, elapsed_seconds=perf_counter()-started))
        if val is None or val < best_loss:
            best_epoch, best_loss, stale = epoch, val, 0
            save_torch(best_path, delta(model, head))
        else:
            stale += 1
        save_torch(last_path, dict(state=delta(model, head), optimizer=optimizer.state_dict(),
                                   epoch=epoch, best_epoch=best_epoch, best_loss=best_loss, stale=stale,
                                   partition_sha256=partition_sha256,
                                   history=history, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
        write_json(destination / 'history.json', history)
        print(f'{destination.name} epoch={epoch} loss={val} best={best_epoch}', flush=True)
    restore(model, head, torch.load(best_path, map_location='cpu', weights_only=True))
    export_heads(model, head)
    model.encoder.eval()
    summary = dict(best_epoch=best_epoch, best_loss=best_loss, best_sha256=digest(best_path),
                   epochs_run=history[-1]['epoch'], elapsed_seconds=perf_counter()-started,
                   partition_sha256=partition_sha256)
    write_json(destination / 'history.json', history)
    write_json(complete, summary)
    last_path.unlink(missing_ok=True)
    return model, summary


def run(output, outer_seed=None, pilot=False):
    started = perf_counter()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    base_path = C.ARTIFACTS / 'e2-dinov3-large'
    base = json.loads((base_path / 'final/model.json').read_text())
    assert digest(base_path / 'encoder.pt') == base['encoder_sha256']
    df = manifest(C.ARTIFACTS / 'audit')
    if outer_seed is not None:
        assignments = inner_folds(df, folds=3, seed=outer_seed, attempts=10000)
        df['fold'] = df.study.map(assignments.set_index('study').fold)
    serialized = pd.read_csv(StringIO(df.to_csv(index=False)))
    pd.testing.assert_frame_equal(serialized.drop(columns='fold'), pd.read_csv(base_path / 'manifest.csv').drop(columns='fold'))
    with np.load(base_path / 'features.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['image_ids'], df.image_id)
        features = stored['features']
    code = [C.ROOT / 'src' / p for p in ('config.py', 'solution/model.py', 'solution/dicom.py',
            'utils/finetune.py', 'utils/train.py', 'utils/train_blend.py', 'utils/evaluate.py', 'utils/data.py')]
    recipe = dict(id='dinov3-last2-bf16-v2', source_encoder_sha256=base['encoder_sha256'], outer_seed=outer_seed,
                  parent_recipe_sha256=digest(base_path/'recipe.json'), features_sha256=digest(base_path/'features.npz'),
                  manifest_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                  code_sha256={str(p.relative_to(C.ROOT)): digest(p) for p in code},
                  learning_rates=[1e-5, 3e-5], warm_head=dict(C=.1, balanced=True, solver='liblinear'),
                  max_epochs=20, patience=4, seed=42, batch=2, effective_batch=16, precision='bf16',
                  quality_gate=True, inner_repeats=1, pilot=pilot,
                  versions=base['training']['versions'], selection='Nested study CV; development estimate')
    if (output/'recipe.json').exists():
        assert json.loads((output/'recipe.json').read_text()) == recipe, 'Recipe changed; use another output'
    write_json(output/'recipe.json', recipe)
    df.to_csv(output/'manifest.csv', index=False)
    for p in code:
        archive = output/'source'/p.relative_to(C.ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, archive)
    splits = {}
    for fold in [0, 1, 2, 'final']:
        train_df = df if fold == 'final' else df[df.fold != fold]
        splits[fold] = inner_folds(train_df.reset_index(drop=True))
        splits[fold].to_csv(output/f'inner_{fold}.csv', index=False)
    images = torch.from_numpy(np.stack([prepare(pydicom.dcmread(C.DATA/p), base['preprocess']) for p in df.path_to_study]))
    print(f'Prepared {len(images)} images; start adaptation', flush=True)
    if pilot:
        if (output/'pilot_checks.json').exists():
            print('Completed pilot checks already saved', flush=True)
            return
        outer = np.flatnonzero(df.fold.to_numpy() != 0)
        assigned = df.study.map(splits[0].set_index('study').fold).to_numpy()
        training = outer[assigned[outer] == 0]
        model, summary = fit_partition(base, df, features, images, training, np.array([], dtype=int),
                                        1e-5, output/'pilot', epochs=2)
        model.save(output/'final')
        before = model.encode(pydicom.dcmread(C.DATA/df.path_to_study.iloc[training[0]]))
        loaded = Model.load(output/'final', 'cuda')
        np.testing.assert_array_equal(before, loaded.encode(pydicom.dcmread(C.DATA/df.path_to_study.iloc[training[0]])))
        # Crash after the first epoch checkpoint, then compare with uninterrupted training.
        original_save = globals()['save_torch']
        def interrupt_after_checkpoint(path, value):
            original_save(path, value)
            if path.name == 'last.pt':
                raise InterruptedError('intentional resume check')
        globals()['save_torch'] = interrupt_after_checkpoint
        try:
            fit_partition(base, df, features, images, training, np.array([], dtype=int),
                          1e-5, output/'interrupted', epochs=2)
            raise AssertionError('Pilot did not interrupt')
        except InterruptedError:
            pass
        finally:
            globals()['save_torch'] = original_save
        resumed, _ = fit_partition(base, df, features, images, training, np.array([], dtype=int),
                                    1e-5, output/'interrupted', epochs=2)
        np.testing.assert_array_equal(before, resumed.encode(pydicom.dcmread(C.DATA/df.path_to_study.iloc[training[0]])))
        assert model.heads == resumed.heads
        write_json(output/'pilot_checks.json', dict(**summary, save_load_exact=True, resume_exact=True,
                                                   peak_vram_bytes=torch.cuda.max_memory_allocated()))
        return
    oof_features = np.empty_like(features)
    for fold in [0, 1, 2, 'final']:
        train = np.arange(len(df)) if fold == 'final' else np.flatnonzero(df.fold.to_numpy() != fold)
        valid = np.array([], dtype=int) if fold == 'final' else np.flatnonzero(df.fold.to_numpy() == fold)
        destination = output/('final' if fold == 'final' else f'fold_{fold}')
        destination.mkdir(exist_ok=True)
        if (destination/'complete.json').exists():
            if fold != 'final':
                oof_features[valid] = np.load(destination/'validation_features.npy')
            continue
        assignments = df.study.map(splits[fold].set_index('study').fold).to_numpy()
        choices = []
        for lr in (1e-5, 3e-5):
            inner_scores = pd.DataFrame(np.nan, index=train, columns=HEADS)
            summaries = []
            for inner_fold in (0, 1):
                inner_train, inner_valid = train[assignments[train] != inner_fold], train[assignments[train] == inner_fold]
                job = output/'jobs'/f'{fold}_lr{lr}_inner{inner_fold}'
                model, summary = fit_partition(base, df, features, images, inner_train, inner_valid, lr, job)
                x = model.features([C.DATA/p for p in df.iloc[inner_valid].path_to_study])
                _, scores = model.classify(x)
                inner_scores.loc[inner_valid, HEADS] = pd.DataFrame(scores, index=inner_valid).reindex(columns=HEADS).to_numpy()
                pd.DataFrame(scores).assign(image_id=df.iloc[inner_valid].image_id.to_numpy()).to_csv(job/'predictions.csv', index=False)
                np.save(job/'validation_features.npy', x)
                write_json(job/'heads.json', model.heads)
                summaries.append(summary)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            choices.append(dict(lr=lr, loss=float(np.mean([s['best_loss'] for s in summaries])),
                                epochs=int(np.floor(np.median([s['best_epoch'] for s in summaries])+.5)),
                                summaries=summaries, scores=inner_scores))
        choice = min(choices, key=lambda c: (c['loss'], c['lr']))
        selected = thresholds(df.iloc[train], choice['scores'])
        job = output/'jobs'/f'{fold}_refit'
        model, summary = fit_partition(base, df, features, images, train, np.array([], dtype=int),
                                        choice['lr'], job, epochs=choice['epochs'])
        for key in HEADS:
            model.heads[key]['threshold'] = selected[key]['threshold']
        model.metadata.update(recipe=recipe['id'], training=recipe, quality_gate=True,
                              source_weights=base['source_weights'],
                              training_partition='all_labeled' if fold == 'final' else f'outer_train_{fold}')
        model.save(destination)
        write_json(destination/'selection.json', dict(parameters=selected, chosen_lr=choice['lr'], epochs=choice['epochs'],
                   candidates=[{k:v for k,v in c.items() if k != 'scores'} for c in choices]))
        inner = choice['scores'].copy()
        inner.index = df.iloc[train].image_id
        inner.to_csv(destination/'inner_predictions.csv', index_label='image_id')
        if fold == 'final':
            inner.to_csv(output/'final_inner_predictions.csv', index_label='image_id')
        else:
            x = model.features([C.DATA/p for p in df.iloc[valid].path_to_study])
            oof_features[valid] = x
            np.save(destination/'validation_features.npy', x)
            predictions(model, df.iloc[valid], x).to_csv(destination/'oof.csv', index=False)
        write_json(destination/'complete.json', dict(encoder_sha256=digest(destination/'encoder.pt'),
                    **({} if fold == 'final' else dict(oof_sha256=digest(destination/'oof.csv')))))
        print(f'Completed outer {fold}: lr={choice["lr"]} epochs={choice["epochs"]}', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(output/'features.npz', features=oof_features, image_ids=df.image_id.to_numpy(dtype=str))
    oof = pd.concat([pd.read_csv(output/f'fold_{fold}/oof.csv') for fold in range(3)], ignore_index=True)
    oof.to_csv(output/'oof.csv', index=False)
    report = evaluate(oof, 2000)
    write_json(output/'metrics.json', report)
    write_json(output/'environment.json', dict(gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, cuda=torch.version.cuda, elapsed_seconds=perf_counter()-started,
               peak_vram_bytes=torch.cuda.max_memory_allocated()))
    write_json(output/'complete.json', dict(recipe_sha256=digest(output/'recipe.json'), metrics_sha256=digest(output/'metrics.json')))
    print(json.dumps(dict(quality=report['metrics']['quality_all'], macro_f1=report['macro_f1'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--outer-seed', type=int)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    run(args.output, args.outer_seed, args.pilot)

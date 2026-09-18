"""Групповые OOF-метрики — качество реального выхода и неопределённость по исследованиям."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config as C


def binary(y, p, decision):
    y, decision = np.asarray(y, dtype=bool), np.asarray(decision, dtype=bool)
    tp, fp = int((y & decision).sum()), int((~y & decision).sum())
    fn, tn = int((y & ~decision).sum()), int((~y & ~decision).sum())
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both = y.any() and not y.all()
    return dict(n=len(y), positives=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                f1_zero_denominator=not bool(2 * tp + fp + fn),
                roc_auc=float(roc_auc_score(y, p)) if both else None,
                pr_auc=float(average_precision_score(y, p)) if y.any() else None,
                sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=(sensitivity + specificity) / 2 if both else None)


def evaluate(df, resamples=2000):
    scopes = {"quality_all": (np.ones(len(df), dtype=bool), "quality")}
    for short, region in (("spine", C.REGION_SPINE), ("hip", C.REGION_FEMUR)):
        scopes[f"quality_{short}"] = ((df.anatomical_region == region).to_numpy(), "quality")
    scopes.update({t: ((df.anatomical_region == r).to_numpy(), t) for t, r in zip(C.TARGETS, C.TARGET_REGIONS)})
    arrays = {}
    for name, (mask, target) in scopes.items():
        arrays[name] = (mask, df[f"true_{target}"].to_numpy(),
                        df[f"prob_{target}"].to_numpy(), df[f"pred_{target}"].to_numpy())

    def score(indices):
        result = {}
        for name, (mask, y, p, decision) in arrays.items():
            ix = indices[mask[indices]]
            result[name] = binary(y[ix], p[ix], decision[ix])
        return result

    point = score(np.arange(len(df)))
    metrics = ["f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy"]
    samples = {name: {m: [] for m in metrics} for name in scopes}
    macro = []
    groups = [np.flatnonzero(df.study.to_numpy() == g) for g in sorted(df.study.unique())]
    rng = np.random.default_rng(42)
    for _ in range(resamples):
        # Индексы с повторениями сохраняют кратность выбранных исследований.
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        scores = score(ix)
        macro.append(np.mean([scores[t]["f1"] for t in C.TARGETS]))
        for name in scopes:
            for metric in metrics:
                value = scores[name][metric]
                if value is not None:
                    samples[name][metric].append(value)
    for name in scopes:
        point[name]["ci95"] = {metric: dict(
            low=float(np.quantile(v, 0.025)) if v else None,
            high=float(np.quantile(v, 0.975)) if v else None, valid_resamples=len(v))
            for metric, v in samples[name].items()}
        mask, y, _, _ = arrays[name]
        point[name]["positive_studies"] = int(df.loc[mask & (y == 1), "study"].nunique())
    return dict(procedure="study-bootstrap-fixed-oof-v1", resamples=resamples, seed=42,
                images=len(df), studies=int(df.study.nunique()), metrics=point,
                region_accuracy=float((df.anatomical_region == df.predicted_region).mean()),
                macro_f1=float(np.mean([point[t]["f1"] for t in C.TARGETS])),
                macro_f1_ci95=[float(np.quantile(macro, q)) for q in (0.025, 0.975)] if macro else None,
                limitation="Development OOF; intervals do not cover model/architecture selection variability.")

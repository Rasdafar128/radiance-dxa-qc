"""Train the full model on the organizers' dataset and write weights + validation report.

Run on the GPU server (not on a laptop):
    python scripts/train.py --data ../data --weights weights --reps 10 --device cuda

Outputs
    weights/region_model.joblib, weights/qc_model.joblib (+ .json with thresholds/metrics)
    reports/metrics.md, reports/metrics.json, reports/oof_predictions.csv, reports/label_noise.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc import __version__, config  # noqa: E402
from dxaqc.features import FeatureExtractor, geometry_frame, task_embeddings  # noqa: E402
from dxaqc.metrics import binary_metrics, cluster_bootstrap  # noqa: E402
from dxaqc.pipeline import blocked_tasks  # noqa: E402
from dxaqc.model import QualityModel, RegionModel, best_f1_threshold, cross_val_oof, logit, sigmoid  # noqa: E402
from dxaqc.region import RegionClassifier  # noqa: E402
from dxaqc.trainset import build_dataset, label_noise_report  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402


def features(u: pd.DataFrame, device: str, weights: Path, cache: Path):
    if cache.exists():
        with open(cache, "rb") as f:
            c = pickle.load(f)
        if c["hashes"] == list(u.folder + "/" + u.hash):
            print("features: embedding cache hit", cache)
            geom, _ = geometry_frame(list(u.pixels), u.region.values, u.side.values)
            return c["raw"], c["flp"], geom
    names = sorted(set(config.EMBEDDERS) | {config.REGION_EMBEDDER})
    local = weights if any((weights / n).exists() or (weights / f"{n}.pt").exists() for n in names) else None
    fx = FeatureExtractor(names, device=device, weights_dir=local)
    imgs = list(u.pixels)
    raw, flp = fx.embed_raw_and_flip(imgs)
    geom, _ = geometry_frame(imgs, u.region.values, u.side.values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(dict(hashes=list(u.folder + "/" + u.hash), raw=raw, flp=flp, geom=geom), f)
    return raw, flp, geom


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]" if v == v else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--weights", default=str(config.WEIGHTS_DIR))
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--cache", default="artifacts/features.pkl")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    weights, reports = Path(a.weights), Path(a.reports)
    reports.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    u = build_dataset(Path(a.data))
    u.drop(columns=["pixels"]).to_csv(reports / "dataset_index.csv", index=False)
    label_noise_report(u).to_csv(reports / "label_noise.csv", index=False)
    print(u.groupby(["region", "side", "labeled"]).size())
    raw, flp, geom = features(u, a.device, weights, Path(a.cache))
    regions, sides = u.region.values, u.side.values
    temb = task_embeddings(raw, flp, regions, sides)

    # ------------------------------------------------------- region classifier
    ids = np.arange(len(u))
    acc = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED).split(u, regions, u.folder):
        clf = RegionClassifier().fit(raw[config.REGION_EMBEDDER][tr], flp[config.REGION_EMBEDDER][tr], regions[tr], ids[tr])
        pr, _, _ = clf.predict(raw[config.REGION_EMBEDDER][te])
        pf, _, _ = clf.predict(flp[config.REGION_EMBEDDER][te])
        acc.append((np.mean(pr == regions[te]) + np.mean(pf == regions[te])) / 2)
    region_clf = RegionClassifier().fit(raw[config.REGION_EMBEDDER], flp[config.REGION_EMBEDDER], regions, ids)
    weights.mkdir(parents=True, exist_ok=True)
    joblib.dump(region_clf, weights / "region_model.joblib")
    report = {"region_classifier_cv_accuracy": float(np.mean(acc))}
    print("region classifier CV accuracy:", report["region_classifier_cv_accuracy"])

    # --------------------------------------------------------- quality models
    qm = QualityModel(meta=dict(version=__version__, trained_at=datetime.now(timezone.utc).isoformat(),
                                reps=a.reps, spec={k: [(i[0], i[1], i[2]) for i in v] for k, v in config.SPEC.items()},
                                context_beta=config.CONTEXT_BETA, encoders=list(config.EMBEDDERS),
                                any_spec={k: [(i[0], i[1], i[2]) for i in v] for k, v in config.ANY_SPEC.items()},
                                any_blend=config.ANY_BLEND))
    oof_rows, per_target = [], []
    for reg in ("spine", "hip"):
        m = ((u.region == reg) & u.labeled).values
        idx = np.where(m)[0]
        sub = u.iloc[idx].reset_index(drop=True)
        g_r = geom.iloc[idx].reset_index(drop=True)
        e_r = {k: v[idx] for k, v in temb.items()}
        Y = sub[[f"y_{t}" for t in config.TASKS[reg]] + ["y_any"]].astype(int)
        oofs = cross_val_oof(reg, g_r, e_r, Y, sub.folder.values, sub.side.values, reps=a.reps, seed=config.SEED)

        th = {}
        for t in config.TASKS[reg]:
            yc = np.concatenate([Y[f"y_{t}"].values] * a.reps)
            pc = np.concatenate([o[t].values for o in oofs])
            th[t], _ = best_f1_threshold(yc, pc)
        # Platt calibration of the noisy-OR "any" probability, then F1-optimal threshold on it
        yc = np.concatenate([Y["y_any"].values] * a.reps)
        pc = np.concatenate([o["any"].values for o in oofs])
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc)[:, None], yc)
        calib = (float(cal.coef_[0, 0]), float(cal.intercept_[0]))
        th["calib"] = calib
        th["any"], _ = best_f1_threshold(yc, sigmoid(calib[0] * logit(pc) + calib[1]))
        qm.thresholds[reg] = th

        avg = sum(o.astype(float) for o in oofs) / a.reps  # rep-averaged OOF probabilities
        avg["any_cal"] = sigmoid(calib[0] * logit(avg["any"]) + calib[1])
        groups = sub.folder.values
        for t in config.TASKS[reg] + ["any"]:
            y = Y[f"y_{t}"].values
            p = avg["any_cal"].values if t == "any" else avg[t].values
            thr = th[t]
            met = binary_metrics(y, p, thr)
            met["roc_auc_ci"] = cluster_bootstrap(lambda i: roc_auc_score(y[i], p[i]), groups, a.n_boot)
            met["f1_ci"] = cluster_bootstrap(lambda i: f1_score(y[i], p[i] >= thr), groups, a.n_boot)
            met["roc_auc_rep_sd"] = float(np.std([roc_auc_score(y, (o[t] if t != "any" else o["any"]).values) for o in oofs]))
            per_target.append(dict(region=reg, target=t, **met))
        for i, r in sub.iterrows():
            probs = {t: float(avg.loc[i, t]) for t in config.TASKS[reg]}
            probs["any"] = float(avg.loc[i, "any"])
            q = QualityModel(thresholds=qm.thresholds)
            gi = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in g_r.iloc[i].items()}
            qc, names, p_any = q.decide(reg, probs, blocked_tasks(reg, gi))
            gt_names = sorted({config.VIOLATION_NAMES[(reg, t)] for t in config.TASKS[reg] if r[f"y_{t}"] == 1})
            oof_rows.append(dict(n=r.n, folder=r.folder, region=reg, side=r.side, y_any=int(r.y_any),
                                 quality_prob=p_any, quality_class=qc, pred_violations=";".join(names),
                                 gt_violations=";".join(gt_names), comment=r.comment,
                                 **{f"p_{t}": probs[t] for t in config.TASKS[reg]},
                                 **{f"y_{t}": int(r[f"y_{t}"]) for t in config.TASKS[reg]}))
        qm.regions[reg] = RegionModel(reg).fit(g_r, e_r, Y)

    # ------------------------------------------------ pooled image-level view
    O = pd.DataFrame(oof_rows)
    O.to_csv(reports / "oof_predictions.csv", index=False)
    groups = O.folder.values
    yb, pb, cb = O.y_any.values, O.quality_prob.values, O.quality_class.values
    # quality_class is already decided per region (own thresholds) -> metrics of the final decision
    pooled = binary_metrics(yb, cb.astype(float), 0.5)
    pooled["roc_auc"], pooled["pr_auc"] = roc_auc_score(yb, pb), average_precision_score(yb, pb)
    pooled["roc_auc_ci"] = cluster_bootstrap(lambda i: roc_auc_score(yb[i], pb[i]), groups, a.n_boot)
    pooled["f1_ci"] = cluster_bootstrap(lambda i: f1_score(yb[i], cb[i]), groups, a.n_boot)
    per_target.append(dict(region="ALL", target="quality_class", **pooled))
    # per official violation name (pooled over regions) and macro-F1
    names = sorted(set(config.VIOLATION_NAMES.values()))
    f1s = {}
    for nm in names:
        yt = O.gt_violations.str.split(";").apply(lambda s: nm in s).astype(int).values
        yp = O.pred_violations.str.split(";").apply(lambda s: nm in s).astype(int).values
        f1s[nm] = float(f1_score(yt, yp, zero_division=0))
    macro = float(np.mean(list(f1s.values())))
    report.update(per_target=per_target, violation_f1=f1s, violation_macro_f1=macro,
                  thresholds=qm.thresholds)
    qm.meta["cv"] = dict(macro_f1=macro, pooled_roc_auc=pooled["roc_auc"], pooled_f1=pooled["f1"])
    qm.save(weights / "qc_model.joblib")

    (reports / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    lines = [f"# Метрики кросс-валидации ({a.reps}× повторённая 5-fold, группировка по исследованию)", "",
             f"Классификатор области: accuracy {report['region_classifier_cv_accuracy']:.3f}", "",
             "| Область | Цель | n | поз. | Чувств. | Специф. | Bal.Acc | F1 [95% ДИ] | ROC-AUC [95% ДИ] | PR-AUC | порог |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_target:
        lines.append(f"| {r['region']} | {r['target']} | {r['n']} | {r['n_pos']} | {r['sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['balanced_accuracy']:.3f} | {fmt_ci(r['f1'], r['f1_ci'])} | "
                     f"{fmt_ci(r['roc_auc'], r['roc_auc_ci'])} | {r['pr_auc']:.3f} | {r['threshold']:.3f} |")
    lines += ["", "## F1 по типам нарушений (официальный словарь)", ""]
    lines += [f"- {k}: {v:.3f}" for k, v in f1s.items()] + [f"- **macro-F1: {macro:.3f}**", ""]
    lines += ["Пороги выбраны по F1 на OOF-предсказаниях, поэтому F1 немного оптимистичен; ROC-AUC от порога не зависит."]
    (reports / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

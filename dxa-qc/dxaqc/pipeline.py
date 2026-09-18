"""Inference engine: DICOM files of one study -> per-file result rows."""
from __future__ import annotations

import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from . import config
from .dicom_io import ImageRecord, read_image
from .features import FeatureExtractor, available_cpus, compute_task_embeddings, geometry_frame, hip_sides
from .model import QualityModel

log = logging.getLogger("dxaqc")

TASK_SHORT = {"ukl": "укладка", "axis": "ось", "art": "посторонние предметы", "rot": "укладка/ротация",
              "roi": "область интереса"}
# Reference limits from the TZ, shown next to the measurement so the verdict is checkable by eye.
LIMIT_AXIS_DEG = 5.0
LIMIT_TOP_MM = 30.0
LIMIT_LAT_MM = 20.0
LIMIT_BELOW_LT_MM = 30.0


@dataclass
class ImageResult:
    region: str
    region_conf: float
    ood: bool
    side: str
    task_probs: dict
    quality_class: int
    quality_prob: float
    violations: list
    geometry: dict
    viz: dict
    details: list = field(default_factory=list)
    verdicts: dict = field(default_factory=dict)  # task -> (норма|нарушение, пояснение)


def _measure(region: str, task: str, g: dict) -> str:
    """Human-readable evidence behind a criterion (empty if nothing was measured)."""
    if region == "spine":
        if task == "axis" and g.get("angle_deg") is not None:
            return f"наклон оси {g['angle_deg']:.1f}° при допуске {LIMIT_AXIS_DEG:.0f}°"
        if task == "ukl":
            found = [v for v in (g.get("crest_h_left"), g.get("crest_h_right")) if v]
            return ("края подвздошных костей в кадре" if found
                    else "края подвздошных костей в кадре не найдены — нижняя граница скана выше нужной")
        if task == "art":
            bright = (g.get("tophat_top_p99") or 0) >= 30
            return "яркие тонкие включения в кадре" if bright else "ярких тонких включений не видно"
    else:
        if not g.get("ok"):
            return "контур бедренной кости не выделен"
        if task == "roi":
            parts = [f"над б. вертелом {g['top_margin_mm']:.0f} мм (норма ≥{LIMIT_TOP_MM:.0f})",
                     f"латерально {g['lat_margin_mm']:.0f} мм (норма ≥{LIMIT_LAT_MM:.0f})"]
            parts.append(f"ниже м. вертела {g['below_lt_mm']:.0f} мм (норма ≥{LIMIT_BELOW_LT_MM:.0f})"
                         if g.get("below_lt_mm") is not None else "уровень м. вертела не определён")
            return ", ".join(parts)
        if task == "rot":
            prom = g.get("lt_prom_sw")
            if g.get("below_lt_mm") is None:
                return ("малый вертел на медиальном контуре не виден — похоже на избыточную (внутреннюю) ротацию бедра")
            if prom is not None and prom > 0.35:
                return (f"малый вертел крупно выступает (выступ {prom:.2f} ширины диафиза) — "
                        "похоже на недостаточную ротацию бедра")
            return f"малый вертел слегка деформирует контур (выступ {prom:.2f} ширины диафиза) — ротация в норме"
    return ""


def _explain(region: str, g: dict, probs: dict, violations: list, blocked: set, thresholds: dict,
             implant: bool, ood: bool):
    """Per-criterion verdict + evidence + probability with its threshold."""
    verdicts, out = {}, []
    for t in config.TASKS[region]:
        bad = config.VIOLATION_NAMES[(region, t)] in violations
        why = _measure(region, t, g)
        verdict = "НАРУШЕНИЕ" if bad else "норма"
        p, th = probs.get(t, float("nan")), thresholds.get(t)
        # «риск» — вероятность нарушения по этому критерию; знак сразу показывает,
        # с какой стороны от порога она оказалась, иначе вердикт читается как противоречие.
        tail = f"риск {p:.2f}" + (f" {'>' if p > th else '≤'} {th:.2f}" if th is not None else "")
        if t in (blocked or set()):
            tail += " (решает измерение)"
        verdicts[t] = (verdict, why, tail, config.CRITERION_RULES.get((region, t), ""))
        out.append(f"{TASK_SHORT[t]}: {verdict}" + (f" — {why}" if why else "") + f", {tail}")
    if g.get("w_px") and g.get("h_px"):
        out.append(f"поле сканирования {g['w_px'] * config.PIXEL_MM:.0f}×{g['h_px'] * config.PIXEL_MM:.0f} мм")
    if implant:
        out.append("ВНИМАНИЕ: признаки металлоконструкции или эндопротеза — область может быть непригодна для денситометрии")
    if ood:
        out.append("ВНИМАНИЕ: изображение нетипично для обучающей выборки, результат требует проверки специалистом")
    return out, verdicts


def blocked_tasks(region: str, g: dict) -> set:
    if region == "spine" and g.get("angle_deg") is not None and g["angle_deg"] < config.AXIS_MIN_ANGLE_DEG:
        return {"axis"}
    return set()


class Engine:
    def __init__(self, weights_dir: str | os.PathLike | None = None, device: str | None = None):
        torch.set_grad_enabled(False)
        torch.manual_seed(config.SEED)
        torch.set_num_threads(int(os.environ.get("DXAQC_THREADS", min(16, available_cpus()))))
        wdir = Path(weights_dir) if weights_dir else config.WEIGHTS_DIR
        self.qc = QualityModel.load(wdir / "qc_model.joblib")
        self.region_clf = joblib.load(wdir / "region_model.joblib")
        names = sorted(set(config.EMBEDDERS) | {config.REGION_EMBEDDER})
        self.fx = FeatureExtractor(names, device=device, weights_dir=wdir)
        self.version = self.qc.meta.get("version", "unknown")

    # ------------------------------------------------------------------ core
    def analyze(self, imgs: list[np.ndarray], study_keys: list[str]) -> list[ImageResult]:
        region_emb = self.fx.embedders[config.REGION_EMBEDDER](imgs)
        regions, conf, ood = self.region_clf.predict(region_emb)
        sides = hip_sides(imgs, regions)
        temb = compute_task_embeddings(self.fx, imgs, regions, sides, {config.REGION_EMBEDDER: region_emb})
        geom, viz = geometry_frame(imgs, regions, sides)
        task_probs: list[dict] = [{} for _ in imgs]
        any_raw = np.zeros(len(imgs))
        for reg in ("spine", "hip"):
            idx = np.where(regions == reg)[0]
            if len(idx) == 0:
                continue
            P = self.qc.regions[reg].predict_tasks(
                geom.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in temb.items()},
                np.asarray(study_keys)[idx], np.asarray(sides)[idx])
            for j, i in enumerate(idx):
                task_probs[i] = {t: float(P.iloc[j][t]) for t in config.TASKS[reg]}
                any_raw[i] = float(P.iloc[j]["any"])
        results = []
        for i, im in enumerate(imgs):
            reg = str(regions[i])
            probs = dict(task_probs[i], any=any_raw[i])
            g = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in geom.iloc[i].items()}
            blocked = blocked_tasks(reg, g)
            qc, names, p_any = self.qc.decide(reg, probs, blocked)
            implant = reg == "hip" and float((im >= 250).mean()) > config.IMPLANT_SAT_FRAC
            details, verdicts = _explain(reg, g, task_probs[i], names, blocked,
                                         self.qc.thresholds.get(reg, {}), implant, bool(ood[i]))
            results.append(ImageResult(
                region=reg, region_conf=float(conf[i]), ood=bool(ood[i]), side=sides[i],
                task_probs=task_probs[i], quality_class=qc, quality_prob=p_any, violations=names,
                geometry=g, viz=viz[i], details=details, verdicts=verdicts))
        return results

    # ----------------------------------------------------------------- study
    def process_study(self, key: str, paths: list[str], root: str | os.PathLike):
        """Returns (rows, records, results_by_hash). Never raises."""
        t0 = time.perf_counter()
        records: list[ImageRecord] = [read_image(p) for p in paths]
        results: dict[str, ImageResult] = {}
        study_error = None
        try:
            uniq: dict[str, ImageRecord] = {}
            for r in records:
                if r.pixels is not None:
                    uniq.setdefault(r.pixel_hash, r)  # byte-identical duplicates share one analysis
            if uniq:
                hashes = list(uniq)
                res = self.analyze([uniq[h].pixels for h in hashes], [key] * len(hashes))
                results = dict(zip(hashes, res))
        except Exception as e:  # noqa: BLE001
            study_error = f"{type(e).__name__}: {e}"
            log.error("study %s failed: %s\n%s", key, study_error, traceback.format_exc())
        elapsed = time.perf_counter() - t0
        if elapsed > config.MAX_STUDY_SECONDS:
            log.warning("study %s took %.1fs (> %.0fs)", key, elapsed, config.MAX_STUDY_SECONDS)
        per_file = elapsed / max(len(records), 1)
        rows = []
        for r in records:
            row = dict(path_to_study=os.path.relpath(r.path, root), study_uid=r.study_uid, image_uid=r.sop_uid,
                       time_of_processing=round(per_file, 3), anatomical_region="", quality_class=None,
                       violation_type="", processing_status="Failure", quality_prob=None, side="",
                       details="", error=r.error or study_error)
            x = results.get(r.pixel_hash) if r.pixels is not None else None
            if x is not None and row["error"] is None:
                row.update(anatomical_region=config.REGION_NAMES[x.region], quality_class=int(x.quality_class),
                           violation_type=config.VIOLATION_SEP.join(x.violations), processing_status="Success",
                           quality_prob=round(x.quality_prob, 4), side=x.side, details=" | ".join(x.details))
            elif row["error"] is None:
                row["error"] = "image was not analysed"
            rows.append(row)
        return rows, records, results

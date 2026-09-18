"""Проверки новых рецептов без оценки меток внешних фолдов.

python -m src.utils.check_research --reference artifacts/sources/medimageinsight/official/davit_v1.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
import torch
from safetensors.torch import load_file

from .. import config as C
from ..solution.dicom import prepare
from ..solution.model import HEADS, Blend, Model, digest, fit_head, preprocessing, probability
from .train import manifest, write_json


def check_gate():
    model = Model.__new__(Model)
    model.metadata = {}
    model.heads = {k: dict(coef=[0.], intercept=2., threshold=.5) for k in HEADS}
    model.heads["region"].update(coef=[3.], intercept=0.)
    model.heads["spine_quality"]["intercept"] = -2.
    x = np.array([[-1.], [1.]])
    before, scores = model.classify(x)
    model.metadata["quality_gate"] = True
    after, gated_scores = model.classify(x)
    assert before.quality_class.tolist() == [1, 1]
    assert after.quality_class.tolist() == [0, 1]
    assert after.violation_type.iloc[0] == ""
    assert after.violation_type.iloc[1] == before.violation_type.iloc[1]
    np.testing.assert_array_equal(before.quality_prob, after.quality_prob)
    assert all(np.array_equal(scores[k], gated_scores[k]) for k in HEADS)
    assert (after.quality_class == after.violation_type.ne("").astype(int)).all()
    model.metadata.update(dimensions=1, preprocess=preprocessing("b0"))
    other = Model.__new__(Model)
    other.metadata = dict(model.metadata)
    other.heads = {k: dict(v) for k, v in model.heads.items()}
    other.heads["hip_quality"]["intercept"] = 0.
    blend = Blend([model, other], {k: dict(threshold=.5) for k in HEADS})
    assert str(blend.device) == "cpu"
    _, combined = blend.classify(np.concatenate([x, x], axis=1))
    _, other_scores = other.classify(x)
    for key in HEADS:
        np.testing.assert_array_equal(combined[key], (scores[key] + other_scores[key]) / 2)


def run(reference, device):
    check_gate()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    x, y = np.zeros((100, 4)), np.r_[np.ones(5), np.zeros(95)]
    scores = {s: float(probability(x, fit_head(x, y, .01, False, s)).mean())
              for s in ("liblinear", "lbfgs")}
    assert abs(scores["lbfgs"] - y.mean()) < 1e-3
    assert scores["liblinear"] > .3

    # Исполняется только проверенная исходная версия; убраны необязательные импорты.
    assert digest(reference) == "a23fb75f0a1da4283004e3bc9abe00cbb1f1093994e0fb7006af1f5159850db0"
    code = reference.read_text()
    for line in ("from .registry import register_image_encoder", "import mup.init",
                 "from mup import MuReadout, set_base_shapes", "@register_image_encoder"):
        code = code.replace(line, "")
    namespace = {}
    exec(compile(code, str(reference), "exec"), namespace)
    weights = C.ARTIFACTS / "pretrained/medimageinsight"
    model = Model(device, pretrained=True, backbone="medimageinsight", weights=weights)
    config = model.metadata["encoder_config"]
    original = namespace["create_encoder"](config).eval().to(device)
    state = load_file(weights / "original.pt")
    original.load_state_dict({k.removeprefix("image_encoder."): v for k, v in state.items()
                              if k.startswith("image_encoder.")}, strict=True)
    projection = state["image_projection"].to(device)
    ds = pydicom.dcmread(C.DATA / manifest(C.ARTIFACTS / "audit").path_to_study.iloc[0])
    tensor = torch.from_numpy(prepare(ds, model.metadata["preprocess"])).unsqueeze(0).to(device)
    with torch.inference_mode():
        expected = torch.nn.functional.normalize(original.forward_features(tensor) @ projection, dim=-1)
        actual = model.encoder(tensor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 1024) and abs(float(actual.norm()) - 1) < 1e-6
    report = dict(intercept_constant_features=scores, medimageinsight_reference_max_delta=0,
                  medimageinsight_parameters=sum(p.numel() for p in model.encoder.parameters()))
    del model, original, state, projection, tensor, expected, actual
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    model = Model(device, pretrained=True, backbone="dinov3-large",
                  weights=C.ARTIFACTS / "pretrained/dinov3-large", pooling="spatial")
    tensor = torch.from_numpy(prepare(ds, model.metadata["preprocess"])).unsqueeze(0).to(device)
    with torch.inference_mode():
        states = model.encoder(tensor).last_hidden_state
        grid = states[:, 1 + model.encoder.config.num_register_tokens:].reshape(1, 28, 28, 1024)
        quadrants = torch.stack([grid[:, a:a+14, b:b+14].mean(dim=(1, 2))
                                for a in (0, 14) for b in (0, 14)], dim=-1).flatten(1)
        expected = torch.cat([states[:, 0], quadrants], dim=1).cpu().numpy()[0]
    actual = model.encode(ds)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (5120,)
    report["spatial_manual_max_delta"] = float(np.max(np.abs(actual - expected)))
    destination = C.ARTIFACTS / "research-cycle-2"
    destination.mkdir(exist_ok=True)
    write_json(destination / "implementation_checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.reference, args.device)

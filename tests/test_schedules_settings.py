import json

import pytest

from neodisco.schedules import parse_schedule
from neodisco.settings import DEFAULTS, effective_record, normalise_settings
from neodisco.benchmark import packaged_representative_config


def valid(**updates):
    data = dict(DEFAULTS, prompts=["one"], weights=[1.0])
    data.update(updates)
    return data


def test_schedule_resamples_and_bounds_expansion():
    assert parse_schedule("[12]*2+[4]*2", 2) == [12, 4]
    with pytest.raises(ValueError, match="10000"):
        parse_schedule("[1]*10001", 10)
    with pytest.raises(ValueError, match="complex"):
        parse_schedule("+".join("[1]" for _ in range(200)), 10)


def test_minimal_original_disco_json_uses_canonical_defaults(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"text_prompts": {"0": ["blue:2"]}}))
    settings = normalise_settings(path)
    assert settings["prompts"] == ["blue"]
    assert settings["weights"] == [1.0]
    assert settings["clip_models"] == DEFAULTS["clip_models"]
    assert settings["cut_overview"] == DEFAULTS["cut_overview"]


def test_internal_settings_and_clip_pairs_roundtrip(tmp_path):
    source = valid(
        prompts=["warm", "cold"], weights=[2.0, -0.5],
        clip_models=[["ViT-B-32-quickgelu", "openai"]],
    )
    first = normalise_settings(source)
    assert first["weights"] == pytest.approx([4 / 3, -1 / 3])
    assert first["clip_models"] == ["ViTB32"]
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(effective_record(first)))
    assert normalise_settings(path)["prompts"] == first["prompts"]


@pytest.mark.parametrize("updates,match", [
    ({"width": 65}, "width"),
    ({"steps": 0}, "steps"),
    ({"skip_steps": 250}, "skip_steps"),
    ({"eta": 1.1}, "eta"),
    ({"weights": [1.0, -1.0], "prompts": ["a", "b"]}, "signed sum"),
    ({"weights": [float("inf")]}, "finite"),
    ({"clip_models": []}, "CLIP"),
    ({"cut_batch": 0}, "cut_batch"),
    ({"seed": -2}, "seed"),
    ({"init_image": "/definitely/missing.png"}, "init_image"),
])
def test_invalid_settings_fail(updates, match):
    with pytest.raises(ValueError, match=match):
        normalise_settings(valid(**updates))


def test_every_step_needs_a_cut():
    with pytest.raises(ValueError, match="every step"):
        normalise_settings(valid(steps=2, cut_overview=[1, 0], cut_innercut=[0, 0],
                                 cut_icgray_p=[0, 0]))


def test_legacy_saved_fp16_alias_migrates():
    old = valid()
    old.pop("precision")
    old["fp16"] = True
    settings = normalise_settings(old)
    assert settings["precision"] == "fp16"


def test_packaged_representative_config_is_available():
    settings = normalise_settings(packaged_representative_config())
    assert (settings["width"], settings["height"], settings["steps"]) == (1280, 768, 250)

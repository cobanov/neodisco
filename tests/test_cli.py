import pytest

from neodisco.cli import build_parser, settings_from_args


def test_legacy_precision_and_attention_aliases_are_preserved():
    args = build_parser().parse_args([
        "a prompt", "--autocast", "none", "--no-fast-attention"])
    settings = settings_from_args(args)
    assert settings["precision"] == "fp32"
    assert settings["attention"] == "original"


def test_retired_strength_fails_before_model_loading():
    args = build_parser().parse_args(["a prompt", "--strength", "3"])
    with pytest.raises(ValueError, match="retired"):
        settings_from_args(args)

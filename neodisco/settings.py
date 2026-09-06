"""Canonical render settings shared by the CLI, benchmark and web API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import disco_config
from .schedules import parse_schedule


SCHEMA_VERSION = 1
DEFAULTS: dict[str, Any] = {
    "prompts": [],
    "weights": [],
    "clip_models": ["ViTB32", "ViTB16", "RN50"],
    "image_size": 512,
    "width": 512,
    "height": 512,
    "steps": 250,
    "skip_steps": 0,
    "seed": 0,
    "eta": 0.8,
    "clamp_max": 0.05,
    "clip_scale": 5000.0,
    "tv_scale": 0.0,
    "range_scale": 150.0,
    "sat_scale": 0.0,
    "cutn_batches": 1,
    "cut_overview": 4,
    "cut_innercut": 16,
    "cut_icgray_p": 0.2,
    "inner_size_pow": 0.5,
    "use_secondary": True,
    "clip_denoised": False,
    "init_image": None,
    "init_scale": 0.0,
    "batch_size": 1,
    "device": "auto",
    "precision": "auto",
    "cut_batch": "auto",
    "attention": "sdpa",
    "compile_mode": "eager",
    "grad_checkpoint": False,
    "augment": True,
    "deterministic": False,
    "finite_check": True,
}

WEB_DEFAULTS = dict(
    DEFAULTS,
    width=1280,
    height=768,
    skip_steps=10,
    seed=-1,
    cutn_batches=4,
    cut_overview="[12]*400+[4]*600",
    cut_innercut="[4]*400+[12]*600",
    cut_icgray_p="[0.2]*400+[0]*600",
    inner_size_pow=1.0,
)

_FLOATS = ("eta", "clamp_max", "clip_scale", "tv_scale", "range_scale", "sat_scale",
           "init_scale")
_INTS = ("image_size", "width", "height", "steps", "skip_steps", "seed",
         "cutn_batches", "batch_size")


def _normalise_clip_model(value: Any) -> str:
    if isinstance(value, str):
        if value in disco_config.CLIP_NAMES:
            return value
        candidate = tuple(value.split(":", 1)) if ":" in value else (value, "openai")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        candidate = tuple(value)
    else:
        raise ValueError(f"invalid CLIP model identifier: {value!r}")
    for checkbox, pair in disco_config.CLIP_NAMES.items():
        if tuple(pair) == candidate:
            return checkbox
    raise ValueError(f"unsupported CLIP model: {candidate[0]}:{candidate[1]}")


def load_mapping(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        data = deepcopy(dict(source))
        if "text_prompts" in data:
            return disco_config.from_mapping(data)
        return data
    with open(source, encoding="utf-8") as handle:
        data = json.load(handle)
    return disco_config.from_mapping(data) if "text_prompts" in data else data


def normalise_settings(
    source: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    defaults: Mapping[str, Any] = DEFAULTS,
    resolve_random_seed: bool = False,
) -> dict[str, Any]:
    settings = deepcopy(dict(defaults))
    loaded: dict[str, Any] = {}
    if source is not None:
        loaded = load_mapping(source)
        settings.update(loaded)
        if "precision" not in loaded and "fp16" in loaded:
            settings["precision"] = "fp16" if loaded["fp16"] else "auto"
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})

    for key in ("clip_models", "cut_overview", "cut_innercut", "cut_icgray_p"):
        if settings.get(key) is None:
            settings[key] = deepcopy(defaults[key])

    for key in _INTS:
        if isinstance(settings[key], bool):
            raise ValueError(f"{key} must be an integer, not a boolean")
        if isinstance(settings[key], float) and not settings[key].is_integer():
            raise ValueError(f"{key} must be an integer")
        settings[key] = int(settings[key])
    for key in _FLOATS:
        settings[key] = float(settings[key])
    settings["prompts"] = [str(p).strip() for p in settings.get("prompts", [])]
    settings["weights"] = [float(w) for w in settings.get("weights", [])]
    if any(not math.isfinite(w) for w in settings["weights"]):
        raise ValueError("prompt weights must be finite")
    weight_sum = sum(settings["weights"])
    if settings["weights"] and abs(weight_sum) < 1e-3:
        raise ValueError("prompt weights must have an absolute signed sum of at least 0.001")
    if settings["weights"]:
        settings["weights"] = [w / abs(weight_sum) for w in settings["weights"]]
    settings["clip_models"] = [_normalise_clip_model(v) for v in settings.get("clip_models", [])]
    for key in ("use_secondary", "clip_denoised", "grad_checkpoint", "augment",
                "deterministic", "finite_check"):
        value = settings[key]
        if isinstance(value, bool):
            settings[key] = value
        elif isinstance(value, int) and value in {0, 1}:
            settings[key] = bool(value)
        else:
            raise ValueError(f"{key} must be a boolean")
    settings["device"] = str(settings["device"])
    settings["precision"] = str(settings["precision"])
    settings["attention"] = str(settings["attention"])
    settings["compile_mode"] = str(settings["compile_mode"])
    if settings["cut_batch"] != "auto":
        settings["cut_batch"] = int(settings["cut_batch"])
    if resolve_random_seed and settings["seed"] < 0:
        settings["seed"] = secrets.randbelow(2**31)
    validate_settings(settings)
    return settings


def validate_settings(s: Mapping[str, Any]) -> None:
    if s["image_size"] not in {256, 512}:
        raise ValueError("image_size must be 256 or 512")
    for axis in ("width", "height"):
        if s[axis] <= 0 or s[axis] % 64:
            raise ValueError(f"{axis} must be a positive multiple of 64")
    if not 1 <= s["steps"] <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if not 0 <= s["skip_steps"] < s["steps"]:
        raise ValueError("skip_steps must be at least 0 and less than steps")
    if s["seed"] < -1 or s["seed"] > 2**63 - 1:
        raise ValueError("seed must be -1 (random) or between 0 and 2^63-1")
    if not 0 <= s["eta"] <= 1:
        raise ValueError("eta must be between 0 and 1")
    for key in _FLOATS:
        if not math.isfinite(s[key]):
            raise ValueError(f"{key} must be finite")
    if s["clamp_max"] < 0 or s["init_scale"] < 0:
        raise ValueError("clamp_max and init_scale must be non-negative")
    for value in parse_schedule(s["inner_size_pow"], 1000):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("inner_size_pow must be finite and positive")
    if s["cutn_batches"] < 1 or s["batch_size"] < 1:
        raise ValueError("cutn_batches and batch_size must be positive")
    if s["cut_batch"] != "auto" and s["cut_batch"] < 1:
        raise ValueError("cut_batch must be auto or a positive integer")
    if not s["prompts"] or any(not p for p in s["prompts"]):
        raise ValueError("at least one non-empty prompt is required")
    if len(s["prompts"]) != len(s["weights"]):
        raise ValueError("prompts and weights must have the same length")
    if not s["clip_models"]:
        raise ValueError("at least one CLIP model is required")
    if s["precision"] not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16 or fp16")
    if s["attention"] not in {"original", "sdpa"}:
        raise ValueError("attention must be original or sdpa")
    if s["compile_mode"] not in {"eager", "default", "reduce-overhead", "max-autotune"}:
        raise ValueError("unsupported compile_mode")
    if not s["finite_check"]:
        raise ValueError("finite_check is mandatory for successful renders")
    schedules = {
        "cut_overview": parse_schedule(s["cut_overview"], 1000),
        "cut_innercut": parse_schedule(s["cut_innercut"], 1000),
        "cut_icgray_p": parse_schedule(s["cut_icgray_p"], 1000),
    }
    for value in schedules["cut_overview"] + schedules["cut_innercut"]:
        if not math.isfinite(float(value)) or int(value) != value or value < 0:
            raise ValueError("cut schedules must contain non-negative integers")
    if not all((o + i) > 0 for o, i in zip(schedules["cut_overview"], schedules["cut_innercut"])):
        raise ValueError("at least one cut is required at every step")
    for value in schedules["cut_icgray_p"]:
        if not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError("cut_icgray_p values must be between 0 and 1")
    if s["init_scale"] and not s.get("init_image"):
        raise ValueError("init_scale requires init_image")
    if isinstance(s.get("init_image"), (str, os.PathLike)) and not Path(s["init_image"]).is_file():
        raise ValueError(f"init_image does not exist: {s['init_image']}")


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def init_reference(path: str | os.PathLike[str] | None) -> dict[str, str] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"init_image does not exist: {path}")
    return {"path": str(path), "sha256": checkpoint_sha256(resolved)}


def effective_record(settings: Mapping[str, Any]) -> dict[str, Any]:
    record = deepcopy(dict(settings))
    record["schema_version"] = SCHEMA_VERSION
    record["sampling_semantics"] = "disco-2026-09-07"
    record["range_gradient"] = "original-upstream-zero"
    record["cutout_resize"] = "ResizeRight-510d4d5"
    record["cutout_draws"] = "independent-per-model"
    record["secondary_precision"] = "fp32"
    record["seed_scope"] = "one-render"
    record["init_image_ref"] = init_reference(record.get("init_image"))
    return record

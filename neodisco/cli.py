"""Command line entry point and shared render construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import disco_config
from .clip_bank import ClipBank
from .cutouts import MakeCutouts
from .guidance import PromptGuidance
from .runtime import auto_cut_batch, prepare_reference_runtime, resolve_runtime
from .settings import DEFAULTS, effective_record, normalise_settings


def _raise_fd_limit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass


def parse_prompts(items):
    prompts, weights = [], []
    for item in items:
        if "::" in item:
            prompt, weight = item.rsplit("::", 1)
            prompts.append(prompt.strip())
            weights.append(float(weight))
        else:
            prompts.append(item.strip())
            weights.append(1.0)
    return prompts, weights


def build_parser():
    ap = argparse.ArgumentParser(description="Run original Disco Diffusion checkpoints")
    ap.add_argument("prompt", nargs="*", help='prompts; weight with "text::0.5"')
    ap.add_argument("--disco-config", help="original Disco or saved neodisco settings JSON")
    ap.add_argument("--image-size", type=int, choices=[256, 512])
    ap.add_argument("--ckpt", help="checkpoint path; defaults to weights/disco")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--skip-steps", type=int)
    ap.add_argument("--init-image")
    ap.add_argument("--init-scale", type=float)
    ap.add_argument("--eta", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--strength", type=float, help="retired; use --clamp-max")
    ap.add_argument("--clip-scale", type=float)
    ap.add_argument("--tv-scale", type=float)
    ap.add_argument("--range-scale", type=float)
    ap.add_argument("--sat-scale", type=float)
    ap.add_argument("--clamp-max", type=float)
    ap.add_argument("--overview-cuts", dest="cut_overview")
    ap.add_argument("--inner-cuts", dest="cut_innercut")
    ap.add_argument("--inner-grey-p", dest="cut_icgray_p")
    ap.add_argument("--inner-size-pow", help="positive scalar or a Disco schedule")
    ap.add_argument("--cutn-batches", type=int)
    ap.add_argument("--cut-batch", default=None, help="auto or a positive cutout chunk size")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--clip-models", help="comma list of Disco names or model:weights pairs")
    ap.add_argument("--clip-denoised", action="store_true")
    ap.add_argument("--secondary", help="path to secondary_model_imagenet_2.pth")
    ap.add_argument("--no-secondary", action="store_true")
    ap.add_argument("--device", default=None, help="auto, cpu, cuda or a specific CUDA device")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--fp16", action="store_true", help="compatibility alias for --precision fp16")
    ap.add_argument("--attention", choices=["original", "sdpa"])
    ap.add_argument("--autocast", choices=["none", "bf16"],
                    help="compatibility alias for --precision fp32/bf16")
    ap.add_argument("--no-fast-attention", action="store_true",
                    help="compatibility alias for --attention original")
    ap.add_argument("--compile", dest="compile_mode",
                    choices=["eager", "default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    return ap


def settings_from_args(args, parser=None):
    if args.strength is not None:
        message = "--strength is retired because its former DDIM formula was invalid; use --clamp-max"
        if parser:
            parser.error(message)
        raise ValueError(message)
    overrides = {}
    for key in (
        "image_size", "width", "height", "batch_size", "steps", "skip_steps", "eta",
        "seed", "clip_scale", "tv_scale", "range_scale", "sat_scale", "clamp_max",
        "cutn_batches", "cut_overview", "cut_innercut", "cut_icgray_p", "inner_size_pow",
        "init_image", "init_scale", "device", "precision", "attention", "compile_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.cut_batch is not None:
        overrides["cut_batch"] = args.cut_batch
    if args.prompt:
        overrides["prompts"], overrides["weights"] = parse_prompts(args.prompt)
    if args.clip_models:
        overrides["clip_models"] = [v.strip() for v in args.clip_models.split(",") if v.strip()]
    if args.no_augment:
        overrides["augment"] = False
    if args.clip_denoised:
        overrides["clip_denoised"] = True
    if args.no_secondary:
        overrides["use_secondary"] = False
    if args.grad_checkpoint:
        overrides["grad_checkpoint"] = True
    if args.deterministic:
        overrides["deterministic"] = True
    if args.fp16:
        if args.precision not in (None, "fp16"):
            raise ValueError("--fp16 conflicts with --precision")
        overrides["precision"] = "fp16"
    if args.autocast:
        alias_precision = "fp32" if args.autocast == "none" else "bf16"
        if overrides.get("precision", alias_precision) != alias_precision:
            raise ValueError("--autocast conflicts with --precision/--fp16")
        overrides["precision"] = alias_precision
    if args.no_fast_attention:
        if overrides.get("attention", "original") != "original":
            raise ValueError("--no-fast-attention conflicts with --attention")
        overrides["attention"] = "original"
    source = args.disco_config if args.disco_config else None
    return normalise_settings(source, overrides, defaults=DEFAULTS, resolve_random_seed=True)


def build_pipeline(settings, ckpt=None, secondary_path=None):
    """Load one reusable model/guidance stack without starting a sample."""
    from .backends.pixel import PixelBackend

    policy = resolve_runtime(settings["device"], settings["precision"])
    effective = dict(settings)
    if effective["deterministic"]:
        prepare_reference_runtime(policy.device)
    effective["device"] = str(policy.device)
    effective["precision"] = policy.precision
    if effective["cut_batch"] == "auto":
        effective["cut_batch"] = auto_cut_batch(policy.device)
    checkpoint = ckpt or PixelBackend.default_path(effective["image_size"])
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"diffusion checkpoint not found: {checkpoint}")
    secondary = secondary_path or PixelBackend.default_secondary_path()
    if effective["use_secondary"] and not Path(secondary).is_file():
        raise FileNotFoundError(
            f"secondary model requested but not found: {secondary}; use --no-secondary to opt out"
        )

    pairs = [disco_config.CLIP_NAMES[name] for name in effective["clip_models"]]
    bank = ClipBank(pairs, device=policy.device)
    cutouts = MakeCutouts(bank.cut_size, inner_size_pow=1.0,
                          augment=effective["augment"])
    guidance = PromptGuidance(
        bank, cutouts, effective["prompts"], effective["weights"],
        clip_scale=effective["clip_scale"], tv_scale=effective["tv_scale"],
        range_scale=effective["range_scale"], sat_scale=effective["sat_scale"],
        clamp_max=effective["clamp_max"])
    backend = PixelBackend(
        checkpoint, image_size=effective["image_size"], device=policy.device,
        fp16=policy.model_fp16, use_checkpoint=effective["grad_checkpoint"],
        secondary_path=secondary if effective["use_secondary"] else None,
        fast_attention=effective["attention"] == "sdpa",
        autocast_dtype=policy.autocast_dtype, compile_mode=effective["compile_mode"])
    return guidance, backend, effective


def sample_pipeline(guidance, backend, effective, progress=True, preview=None):
    pixels = backend.sample(
        guidance=guidance, batch_size=effective["batch_size"], steps=effective["steps"],
        seed=effective["seed"], cut_batch=effective["cut_batch"], eta=effective["eta"],
        width=effective["width"], height=effective["height"],
        cut_overview=effective["cut_overview"], cut_innercut=effective["cut_innercut"],
        cut_ic_pow=effective["inner_size_pow"],
        cut_icgray_p=effective["cut_icgray_p"], cutn_batches=effective["cutn_batches"],
        clip_denoised=effective["clip_denoised"], skip_steps=effective["skip_steps"],
        use_secondary=effective["use_secondary"], init_image=effective["init_image"],
        init_scale=effective["init_scale"], progress=progress, preview=preview,
        deterministic=effective["deterministic"], finite_check=effective["finite_check"])
    effective["compile_mode_requested"] = effective["compile_mode"]
    effective["compile_mode_effective"] = backend.effective_compile_mode
    effective["actual_steps"] = effective["steps"] - effective["skip_steps"]
    effective["guidance_nan_steps"] = list(getattr(backend, "guidance_nan_steps", []))
    return pixels


def render(settings, ckpt=None, secondary_path=None, progress=True, preview=None):
    """Load the requested model stack, sample once and return effective settings."""
    guidance, backend, effective = build_pipeline(settings, ckpt, secondary_path)
    pixels = sample_pipeline(guidance, backend, effective, progress, preview)
    return pixels, effective, backend


def save_result(pixels, path, effective, backend):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = backend.to_uint8(pixels)
    image = images[0] if len(images) == 1 else np.concatenate(list(images), axis=1)
    Image.fromarray(image).save(path)
    record = effective_record(effective)
    record["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _raise_fd_limit()
    try:
        settings = settings_from_args(args, parser)
        pixels, effective, backend = render(
            settings, ckpt=args.ckpt, secondary_path=args.secondary, progress=True)
        save_result(pixels, args.out, effective, backend)
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()

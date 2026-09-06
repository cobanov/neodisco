"""Reproducible real-checkpoint benchmark runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.resources
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .cli import _raise_fd_limit, build_pipeline, parse_prompts, sample_pipeline
from .losses import spherical_dist_loss
from .settings import DEFAULTS, checkpoint_sha256, normalise_settings


ROOT = Path(__file__).resolve().parents[1]


def packaged_representative_config():
    text = importlib.resources.files("neodisco").joinpath(
        "data/representative.json").read_text(encoding="utf-8")
    return json.loads(text)


def _source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "neodisco").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance():
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True))
        return {"revision": revision, "dirty": dirty, "source_sha256": _source_fingerprint()}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "source_sha256": _source_fingerprint()}


def environment(device):
    packages = {}
    for name in ("torch", "torchvision", "open_clip_torch", "neodisco"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": (torch.backends.cuda.matmul.allow_tf32
                                        if torch.cuda.is_available() else None),
            "cudnn_allow_tf32": (torch.backends.cudnn.allow_tf32
                                  if torch.cuda.is_available() else None),
        },
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        out["gpu"] = {
            "index": index,
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": props.total_memory,
        }
        try:
            line = subprocess.check_output([
                "nvidia-smi", f"--id={index}", "--query-gpu=driver_version",
                "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
            out["driver"] = line
        except (OSError, subprocess.CalledProcessError, IndexError):
            out["driver"] = None
    return out


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_sample(guidance, backend, effective):
    sync(backend.device)
    started = time.perf_counter()
    pixels = sample_pipeline(guidance, backend, effective, progress=False)
    sync(backend.device)
    return pixels, time.perf_counter() - started


def save_png(pixels, path, backend):
    images = backend.to_uint8(pixels)
    image = images[0] if len(images) == 1 else np.concatenate(list(images), axis=1)
    Image.fromarray(image).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluation_clip_distance(pixels, guidance):
    image = pixels.add(1).div(2).float()
    embedding = guidance.bank.encode_cutouts(image, 0)
    distances = spherical_dist_loss(
        embedding.unsqueeze(1), guidance.embeddings[0].unsqueeze(0))
    return float((distances * guidance.weights).sum(dim=1).mean().cpu())


def profile_sample(guidance, backend, effective, out_dir, name):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if backend.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace = out_dir / f"{name}-trace.json"
    with torch.profiler.profile(activities=activities) as profiler:
        pixels, duration = timed_sample(guidance, backend, effective)
    profiler.export_chrome_trace(str(trace))
    stages = {}
    for event in profiler.key_averages():
        if event.key.startswith("neodisco."):
            stages[event.key] = {
                "calls": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "device_time_total_us": getattr(
                    event, "device_time_total", getattr(event, "self_cuda_time_total", 0)),
            }
    return pixels, {"render_seconds": duration, "trace": trace.name, "stages": stages}


def variant_overrides(name, args):
    common = {
        "device": args.device,
        "precision": args.precision,
        "attention": args.attention,
        "compile_mode": args.compile_mode,
        "cut_batch": args.cut_batch,
        "deterministic": args.deterministic,
    }
    if name == "corrected-eager":
        common["compile_mode"] = "eager"
    elif name == "original-fp32":
        common.update(precision="fp32", attention="original", compile_mode="eager")
    elif name == "compiled":
        common["compile_mode"] = args.compile_mode if args.compile_mode != "eager" else "default"
    return common


def safe_settings(settings):
    result = dict(settings)
    init_path = result.pop("init_image", None)
    if init_path:
        result["init_image_ref"] = {
            "name": Path(init_path).name,
            "sha256": checkpoint_sha256(init_path),
        }
    return result


def run_variant(name, base, args, out_dir):
    settings = normalise_settings(base, variant_overrides(name, args), resolve_random_seed=False)
    checkpoint = Path(args.weights) / {
        256: "256x256_diffusion_uncond.pt",
        512: "512x512_diffusion_uncond_finetune_008100.pt",
    }[settings["image_size"]]
    secondary = Path(args.weights) / "secondary_model_imagenet_2.pth"
    load_started = time.perf_counter()
    guidance, backend, effective = build_pipeline(settings, checkpoint, secondary)
    sync(backend.device)
    load_seconds = time.perf_counter() - load_started
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)

    first_pixels, first_seconds = timed_sample(guidance, backend, effective)
    first_memory = None
    if backend.device.type == "cuda":
        first_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(backend.device),
        }
    first_path = out_dir / f"{name}-first.png"
    first_sha256 = save_png(first_pixels, first_path, backend)
    warmup_seconds = []
    for _ in range(args.warmup):
        _, duration = timed_sample(guidance, backend, effective)
        warmup_seconds.append(duration)
    profile = None
    if args.profile_stages:
        _, profile = profile_sample(guidance, backend, effective, out_dir, name)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    measured_seconds, images = [], []
    for run in range(1, args.runs + 1):
        pixels, duration = timed_sample(guidance, backend, effective)
        measured_seconds.append(duration)
        path = out_dir / f"{name}-run-{run}.png"
        image_sha256 = save_png(pixels, path, backend)
        images.append({"file": path.name, "sha256": image_sha256})
    memory = None
    if backend.device.type == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(backend.device),
        }
    effective["compile_mode_effective"] = backend.effective_compile_mode
    return {
        "name": name,
        "status": "success",
        "checkpoint": {"name": checkpoint.name, "sha256": checkpoint_sha256(checkpoint)},
        "secondary": ({"name": secondary.name, "sha256": checkpoint_sha256(secondary)}
                      if settings["use_secondary"] else None),
        "settings": safe_settings(effective),
        "actual_steps": effective["steps"] - effective["skip_steps"],
        "timing_seconds": {
            "loading": load_seconds,
            "compile_or_first_render": first_seconds,
            "warmup": warmup_seconds,
            "measured": measured_seconds,
            "median_measured": statistics.median(measured_seconds),
        },
        "memory": {"first_render": first_memory, "steady_state": memory},
        "profile": profile,
        "evaluation": {
            "clip_model": settings["clip_models"][0],
            "weighted_spherical_distance": evaluation_clip_distance(pixels, guidance),
        },
        "first_image": {"file": first_path.name, "sha256": first_sha256},
        "measured_images": images,
    }


def build_parser():
    ap = argparse.ArgumentParser(description="Benchmark corrected neodisco renders")
    ap.add_argument("--profile", choices=["smoke", "representative"], default="smoke")
    ap.add_argument("--config", help="Disco or saved neodisco JSON")
    ap.add_argument("--prompt", action="append", dest="prompts",
                    help='override prompts; repeat and weight with "text::0.5"')
    ap.add_argument("--seed", type=int, help="override the config seed")
    ap.add_argument("--weights", default="weights/disco")
    ap.add_argument("--out-dir", default="benchmark-results")
    ap.add_argument("--variant", action="append", choices=[
        "corrected-eager", "original-fp32", "compiled"], dest="variants")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    ap.add_argument("--attention", choices=["original", "sdpa"], default="sdpa")
    ap.add_argument("--compile-mode", choices=["eager", "default", "reduce-overhead", "max-autotune"], default="eager")
    ap.add_argument("--cut-batch", default="auto")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--profile-stages", action="store_true",
                    help="record named stage totals and a Chrome trace for one extra render")
    return ap


def image_comparisons(variants, out_dir):
    if len(variants) < 2:
        return []
    reference = variants[0]
    ref_path = out_dir / reference["measured_images"][0]["file"]
    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255
    comparisons = []
    for variant in variants[1:]:
        candidate_path = out_dir / variant["measured_images"][0]["file"]
        candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.float32) / 255
        delta = candidate - ref
        mse = float(np.mean(delta * delta))
        comparisons.append({
            "reference": reference["name"], "candidate": variant["name"],
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "root_mean_square_error": mse ** 0.5,
            "max_absolute_error": float(np.max(np.abs(delta))),
            "psnr_db": (float("inf") if mse == 0 else float(-10 * np.log10(mse))),
            "identical_sha256": (reference["measured_images"][0]["sha256"] ==
                                  variant["measured_images"][0]["sha256"]),
        })
    return comparisons


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.variants = args.variants or ["corrected-eager"]
    args.runs = args.runs if args.runs is not None else (1 if args.profile == "smoke" else 3)
    args.warmup = args.warmup if args.warmup is not None else (0 if args.profile == "smoke" else 1)
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("runs must be positive and warmup must be non-negative")
    source = args.config
    _raise_fd_limit()
    if source:
        base = normalise_settings(source, defaults=DEFAULTS)
    elif args.profile == "representative":
        base = normalise_settings(packaged_representative_config(), defaults=DEFAULTS)
    else:
        base = normalise_settings(DEFAULTS, {
            "prompts": ["a blue glass sphere"], "weights": [1.0],
            "image_size": 256, "width": 256, "height": 256, "steps": 4,
            "skip_steps": 0, "clip_models": ["ViTB32"], "cut_overview": 1,
            "cut_innercut": 1, "cutn_batches": 1, "cut_batch": 2,
        })
    quality_overrides = {}
    if args.prompts:
        quality_overrides["prompts"], quality_overrides["weights"] = parse_prompts(args.prompts)
    if args.seed is not None:
        quality_overrides["seed"] = args.seed
    if quality_overrides:
        base = normalise_settings(base, quality_overrides)
    if base["seed"] < 0:
        base = normalise_settings(base, resolve_random_seed=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "profile": args.profile,
        "provenance": provenance(),
        "variants": [],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    exit_code = 0
    try:
        device = torch.device(args.device if args.device != "auto" else
                              ("cuda" if torch.cuda.is_available() else "cpu"))
        manifest["environment"] = environment(device)
        for variant in args.variants:
            try:
                result = run_variant(variant, base, args, out_dir)
                manifest["variants"].append(result)
            except Exception as exc:
                manifest["variants"].append({
                    "name": variant, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        manifest["image_comparisons"] = image_comparisons(manifest["variants"], out_dir)
        manifest["status"] = "success"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

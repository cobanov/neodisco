"""Runtime policy for devices, precision, reproducibility and compilation."""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass

import torch


_RNG_LOCK = threading.RLock()
_CUBLAS_DETERMINISTIC_CONFIGS = {":4096:8", ":16:8"}


@dataclass(frozen=True)
class RuntimePolicy:
    device: torch.device
    precision: str
    autocast_dtype: torch.dtype | None
    model_fp16: bool


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def resolve_runtime(device: str = "auto", precision: str = "auto") -> RuntimePolicy:
    resolved = resolve_device(device)
    bf16_supported = False
    if resolved.type == "cuda":
        with torch.cuda.device(resolved):
            bf16_supported = torch.cuda.is_bf16_supported()
    if precision == "auto":
        precision = (
            "bf16"
            if resolved.type == "cuda" and bf16_supported
            else "fp32"
        )
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"precision must be auto, fp32, bf16 or fp16, got {precision!r}")
    if precision in {"bf16", "fp16"} and resolved.type != "cuda":
        raise ValueError(f"{precision} rendering requires a CUDA device")
    if precision == "bf16" and not bf16_supported:
        raise ValueError("bf16 was requested, but this CUDA device does not support it")
    dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return RuntimePolicy(resolved, precision, dtype, precision == "fp16")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        if device.type in {"cpu", "cuda", "xpu", "mps"}:
            return torch.autocast(device_type=device.type, enabled=False)
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def fp32_context(device: torch.device):
    """Disable an enclosing AMP region for numerically sensitive guidance work."""
    if device.type in {"cpu", "cuda", "xpu", "mps"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def prepare_reference_runtime(device: torch.device):
    """Configure strict CUDA libraries before models create cuBLAS handles."""
    if device.type == "cuda":
        value = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if value not in _CUBLAS_DETERMINISTIC_CONFIGS:
            raise ValueError(
                "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for reference mode")


@contextlib.contextmanager
def render_rng(seed: int, device: torch.device, deterministic: bool = False):
    """Own every torch RNG draw for a serialized render, then restore caller state."""
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    if (deterministic and device.type == "cuda" and
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_DETERMINISTIC_CONFIGS):
        raise RuntimeError(
            "reference mode must be prepared before CUDA model loading; use the neodisco "
            "CLI or call prepare_reference_runtime(device) first")
    # fork_rng manipulates process-global generator state. Serializing the context keeps
    # library callers safe as well as the single-worker web runner.
    with _RNG_LOCK:
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        changed_determinism = deterministic and (
            not previous_deterministic or previous_warn_only)
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.random.default_generator.manual_seed(int(seed))
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            if changed_determinism:
                torch.use_deterministic_algorithms(True, warn_only=False)
            try:
                yield
            finally:
                if changed_determinism:
                    torch.use_deterministic_algorithms(
                        previous_deterministic, warn_only=previous_warn_only)


def auto_cut_batch(device: torch.device) -> int:
    """Choose a conservative cutout chunk from currently available CUDA memory."""
    if device.type != "cuda":
        return 8
    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 2**30
    if free_gib < 10:
        return 8
    if free_gib < 16:
        return 16
    if free_gib < 24:
        return 32
    return 64

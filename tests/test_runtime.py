import torch
import pytest

from neodisco.runtime import autocast_context, render_rng, resolve_runtime


def test_render_rng_repeats_and_restores_cpu_state():
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    with render_rng(7, torch.device("cpu")):
        first = torch.randn(5)
    assert torch.equal(torch.random.get_rng_state(), before)
    with render_rng(7, torch.device("cpu")):
        second = torch.randn(5)
    assert torch.equal(first, second)


def test_cpu_precision_validation():
    assert resolve_runtime("cpu", "auto").precision == "fp32"
    with pytest.raises(ValueError, match="requires a CUDA"):
        resolve_runtime("cpu", "bf16")


def test_fp32_runtime_disables_an_outer_autocast():
    layer = torch.nn.Linear(4, 4).float()
    x = torch.randn(2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with autocast_context(torch.device("cpu"), None):
            output = layer(x)
    assert output.dtype == torch.float32


def test_reference_rng_enforces_strict_mode_then_restores_warn_only():
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with render_rng(3, torch.device("cpu"), deterministic=True):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)

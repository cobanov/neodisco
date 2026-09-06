from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import neodisco.backends.pixel as pixel_module
from neodisco.backends.pixel import PixelBackend


class TinyUNet(nn.Module):
    def forward(self, x, _t):
        return torch.cat([x * 0.05, torch.zeros_like(x)], dim=1)


def toy_backend(model=None):
    backend = PixelBackend.__new__(PixelBackend)
    backend.device = torch.device("cpu")
    backend.image_size = 64
    backend.model = model or TinyUNet()
    backend._compiled_model = backend.model
    backend.secondary = None
    backend.autocast_dtype = None
    backend.attention_mode = "original"
    backend.compile_mode = "eager"
    backend.effective_compile_mode = "eager"
    backend.fp16 = False
    return backend


def sample(backend, seed, **kwargs):
    return backend.sample(
        batch_size=1, steps=4, seed=seed, eta=0.8, width=64, height=64,
        progress=False, **kwargs)


def test_sampler_same_seed_repeats_different_seed_changes_and_restores_rng():
    backend = toy_backend()
    torch.manual_seed(999)
    state = torch.random.get_rng_state().clone()
    first = sample(backend, 12)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = sample(backend, 12)
    third = sample(backend, 13)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_eta_and_init_image_repeat():
    backend = toy_backend()
    init = torch.linspace(-1, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    first = sample(backend, 4, init_image=init, skip_steps=1)
    second = sample(backend, 4, init_image=init, skip_steps=1)
    assert torch.equal(first, second)


class InitRecorder:
    def __init__(self):
        self.calls = []
        self.clamp_max = 0

    def set_init(self, init, scale):
        self.calls.append((init, scale))

    def image_gradient(self, pixels, **kwargs):
        return torch.zeros_like(pixels)

    def clamp(self, grad):
        return grad


class PixelLossGuidance(InitRecorder):
    def image_gradient(self, pixels, **kwargs):
        return pixels * 0.1


class TinySecondary(nn.Module):
    def forward(self, x, _t):
        return SimpleNamespace(pred=x * 0.7)


def test_guidance_init_state_is_cleared_between_renders():
    backend = toy_backend()
    guide = InitRecorder()
    init = torch.zeros(1, 3, 64, 64)
    sample(backend, 1, guidance=guide, init_image=init, init_scale=0)
    sample(backend, 1, guidance=guide)
    assert guide.calls[0] == (None, 0.0)
    assert guide.calls[-1] == (None, 0.0)


@pytest.mark.parametrize("secondary", [False, True])
def test_guidance_backward_paths_with_and_without_secondary_are_finite(secondary):
    backend = toy_backend()
    if secondary:
        backend.secondary = TinySecondary()
    result = sample(backend, 3, guidance=PixelLossGuidance(), use_secondary=secondary)
    assert torch.isfinite(result).all()


class NaNUNet(nn.Module):
    def forward(self, x, _t):
        return torch.full((x.shape[0], 6, *x.shape[2:]), float("nan"))


def test_nonfinite_sample_fails_with_step_context():
    with pytest.raises(FloatingPointError, match="non-finite.*step"):
        sample(toy_backend(NaNUNet()), 1)


def test_retired_sampler_options_are_rejected():
    backend = toy_backend()
    with pytest.raises(ValueError, match="retired"):
        sample(backend, 1, guidance_strength=3)
    with pytest.raises(ValueError, match="through_model=False"):
        sample(backend, 1, through_model=False)


class FakeLargeUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def convert_to_fp16(self):
        return self

    def forward(self, x, _t):
        return torch.cat([x * 0 + self.weight, torch.zeros_like(x)], 1)


def test_compile_setup_failure_falls_back_to_eager(monkeypatch, tmp_path):
    monkeypatch.setattr(pixel_module, "UNetModel", FakeLargeUNet)
    monkeypatch.setattr(pixel_module.torch, "load", lambda *args, **kwargs: {"weight": torch.zeros(())})
    monkeypatch.setattr(pixel_module._fast_attention, "configure", lambda *args, **kwargs: None)
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"x")

    def fail(*args, **kwargs):
        raise RuntimeError("controlled compiler failure")

    with pytest.warns(RuntimeWarning, match="using eager"):
        backend = PixelBackend(ckpt, image_size=256, device="cpu", compile_mode="default",
                               compiler=fail)
    assert backend.effective_compile_mode == "eager-fallback"


def test_compile_setup_does_not_hide_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(pixel_module, "UNetModel", FakeLargeUNet)
    monkeypatch.setattr(pixel_module.torch, "load", lambda *args, **kwargs: {"weight": torch.zeros(())})
    monkeypatch.setattr(pixel_module._fast_attention, "configure", lambda *args, **kwargs: None)
    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"x")

    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("controlled OOM")

    with pytest.raises(torch.OutOfMemoryError):
        PixelBackend(ckpt, image_size=256, device="cpu", compile_mode="default", compiler=oom)


def compile_failure_type():
    return type("BackendCompilerFailed", (RuntimeError,), {"__module__": "torch._dynamo.exc"})


def test_lazy_compile_failure_falls_back_to_eager():
    backend = toy_backend()
    failure = compile_failure_type()

    class BrokenCompiled:
        def __call__(self, *args, **kwargs):
            raise failure("lazy failure")

    backend._compiled_model = BrokenCompiled()
    backend.effective_compile_mode = "default"
    x = torch.randn(1, 3, 4, 4)
    with pytest.warns(RuntimeWarning, match="execution failed"):
        result = backend._model_forward(x, torch.tensor([1]))
    assert torch.isfinite(result).all()
    assert backend.effective_compile_mode == "eager-fallback"


def test_lazy_compile_wrapper_does_not_hide_nested_oom():
    backend = toy_backend()
    failure = compile_failure_type()
    try:
        raise torch.OutOfMemoryError("nested OOM")
    except torch.OutOfMemoryError as cause:
        wrapped = failure("compile wrapper")
        wrapped.__cause__ = cause

    class BrokenCompiled:
        def __call__(self, *args, **kwargs):
            raise wrapped

    backend._compiled_model = BrokenCompiled()
    with pytest.raises(failure, match="compile wrapper"):
        backend._model_forward(torch.randn(1, 3, 4, 4), torch.tensor([1]))
    assert backend.effective_compile_mode == "eager"

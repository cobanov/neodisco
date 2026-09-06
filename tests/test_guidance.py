from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from neodisco.clip_bank import ClipBank
from neodisco.guidance import PromptGuidance


class Encoder(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        matrix = torch.tensor([
            [1.0, 0.2, -0.4, 0.7],
            [-0.3, 0.9, 0.5, 0.1],
            [0.6, -0.8, 0.3, 1.1],
        ]) + offset
        self.weight = nn.Parameter(matrix, requires_grad=False)
        self.seen_dtype = None

    def encode_image(self, x):
        self.seen_dtype = x.dtype
        return x.mean((2, 3)) @ self.weight


def toy_bank():
    bank = ClipBank.__new__(ClipBank)
    nn.Module.__init__(bank)
    bank.device = torch.device("cpu")
    bank.models = [Encoder(), Encoder(0.15)]
    bank.means = [torch.zeros(1, 3, 1, 1) for _ in bank.models]
    bank.stds = [torch.ones(1, 3, 1, 1) for _ in bank.models]
    bank.sizes = [4, 4]
    bank.dtype = torch.float32
    return bank


class FixedCuts(nn.Module):
    n_cuts = 2

    def forward(self, pixels, **kwargs):
        return torch.cat([pixels, pixels.flip(-1)])


class CountingLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, target):
        self.calls += 1
        return (x - target).square().mean((1, 2, 3), keepdim=True)


def guidance(init_scale=0):
    bank = toy_bank()
    guide = PromptGuidance.__new__(PromptGuidance)
    guide.bank = bank
    guide.cutouts = FixedCuts()
    guide.clip_scale = 2
    guide.tv_scale = 0
    guide.range_scale = 0
    guide.sat_scale = 0
    guide.clamp_max = 0
    guide.init = None
    guide.init_scale = 0
    guide._lpips = None
    guide.embeddings = [
        torch.tensor([[0.2, 0.4, 0.6, 0.8]]),
        torch.tensor([[0.7, -0.1, 0.2, 0.4]]),
    ]
    guide.weights = torch.tensor([1.0])
    if init_scale:
        model = CountingLPIPS()
        guide.set_init(torch.zeros(1, 3, 4, 4), init_scale, perceptual_model=model)
        return guide, model
    return guide


def test_clip_operations_are_fp32_under_outer_autocast():
    bank = toy_bank()
    x = torch.rand(2, 3, 4, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = bank.encode_cutouts(x, 0)
    assert bank.models[0].seen_dtype == torch.float32
    assert out.dtype == torch.float32
    out.sum().backward()
    assert x.grad.dtype == torch.float32


def test_clip_bank_rejects_non_fp32(monkeypatch):
    with pytest.raises(ValueError, match="fixed to fp32"):
        ClipBank([], device="cpu", dtype=torch.bfloat16)


def test_chunked_gradient_matches_full_and_loss_gradient():
    guide = guidance()
    pixels = torch.linspace(-0.8, 0.9, 48).reshape(1, 3, 4, 4)
    full = guide.image_gradient(pixels, cut_batch=0)
    chunked = guide.image_gradient(pixels, cut_batch=1)
    probe = pixels.clone().requires_grad_(True)
    loss_grad = torch.autograd.grad(guide.loss(probe), probe)[0]
    assert torch.allclose(full, chunked, atol=1e-6, rtol=1e-5)
    assert torch.allclose(full, loss_grad, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(full).all() and full.abs().sum() > 0


def test_lpips_changes_gradient_once_independent_of_chunks():
    plain = guidance().image_gradient(torch.full((1, 3, 4, 4), 0.25), cut_batch=1)
    guide, lpips = guidance(init_scale=3)
    changed = guide.image_gradient(
        torch.full((1, 3, 4, 4), 0.25), cut_batch=1, cutn_batches=2)
    assert not torch.allclose(plain, changed)
    assert lpips.calls == 1


def test_missing_perceptual_model_cannot_be_silent():
    guide = guidance()
    guide.init = torch.zeros(1, 3, 4, 4)
    guide.init_scale = 1
    with pytest.raises(RuntimeError, match="unavailable"):
        guide._perceptual_term(torch.zeros(1, 3, 4, 4))


def test_nonfinite_gradient_is_rejected():
    guide = guidance()
    with pytest.raises(FloatingPointError, match="non-finite"):
        guide.clamp(torch.tensor([float("nan")]))


def test_overflow_safe_clamp():
    guide = guidance()
    guide.clamp_max = 0.05
    result = guide.clamp(torch.tensor([[[[3e30, -3e30]]]]))
    assert torch.isfinite(result).all()
    assert result.square().mean().sqrt() == pytest.approx(0.05)


def test_real_cutouts_have_nonzero_differentiable_gradient():
    from neodisco.cutouts import MakeCutouts

    guide = guidance()
    guide.cutouts = MakeCutouts(4, overview=1, inner=1, augment=False)
    pixels = torch.linspace(-0.9, 0.8, 3 * 8 * 8).reshape(1, 3, 8, 8)
    grad = guide.image_gradient(pixels, cut_batch=1)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

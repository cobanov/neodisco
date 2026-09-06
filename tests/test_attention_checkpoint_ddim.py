import torch
import torch.nn as nn

from neodisco.backends import _fast_attention
from neodisco.backends._guided_diffusion import gaussian_diffusion as gd
from neodisco.backends._guided_diffusion.nn import checkpoint
from neodisco.backends._guided_diffusion.unet import QKVAttentionLegacy
from neodisco.backends.pixel import build_diffusion


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = QKVAttentionLegacy(2)


def test_attention_is_instance_scoped_and_backward_equivalent():
    original, fast = Holder(), Holder()
    _fast_attention.configure(fast, True)
    q1 = torch.randn(2, 24, 7, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    y1, y2 = original.attn(q1), fast.attn(q2)
    assert torch.allclose(y1, y2, atol=2e-6, rtol=2e-5)
    y1.square().sum().backward()
    y2.square().sum().backward()
    assert torch.allclose(q1.grad, q2.grad, atol=3e-6, rtol=3e-5)
    assert "_legacy_forward" not in original.attn.forward.__qualname__


def test_maintained_checkpoint_matches_eager_input_gradient():
    weight = torch.randn(4, 4)
    x1 = torch.randn(3, 4, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    fn = lambda x: torch.sin(x @ weight).square()
    checkpoint(fn, (x1,), (), False).sum().backward()
    checkpoint(fn, (x2,), (), True).sum().backward()
    assert torch.allclose(x1.grad, x2.grad)


def test_toy_ddim_matches_reference_equation():
    diffusion = build_diffusion("ddim4")

    def model(x, _t):
        return torch.cat([torch.zeros_like(x), torch.zeros_like(x)], dim=1)

    x = torch.randn(1, 3, 4, 4)
    t = torch.tensor([2])
    p = diffusion.p_mean_variance(model, x, t, clip_denoised=False)
    alpha = torch.tensor(diffusion.alphas_cumprod[2], dtype=x.dtype)
    previous = torch.tensor(diffusion.alphas_cumprod_prev[2], dtype=x.dtype)
    eps = (x - alpha.sqrt() * p["pred_xstart"]) / (1 - alpha).sqrt()
    expected = previous.sqrt() * p["pred_xstart"] + (1 - previous).sqrt() * eps
    actual = diffusion.ddim_sample(model, x, t, clip_denoised=False, eta=0)["sample"]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

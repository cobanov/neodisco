"""Fused attention for the vendored guided-diffusion UNet.

The 2021 code computes attention as two einsums with an explicit softmax over the full
(tokens x tokens) matrix. At 512x512 that is fine. At 1280x768 the coarsest attention
layer sees several thousand tokens and the explicit matrix is both the slowest part of
the step and the reason fp16 overflows. `scaled_dot_product_attention` does the same
arithmetic with a fused kernel and a numerically stable softmax.

The vendored files are left untouched. Each backend configures only its own attention
module instances, so an SDPA backend cannot contaminate a later reference backend.
"""

import types

import torch
import torch.nn.functional as F

from ._guided_diffusion import unet as _unet


def _sdpa(q, k, v, bs, heads, ch, length):
    """q, k, v: (bs*heads, ch, length). SDPA wants (bs, heads, length, ch), contiguous.

    The layout matters: with 3-D or non-contiguous inputs PyTorch silently takes the
    unfused "math" path, which is slower than the original einsums. Shaped like this it
    dispatches to flash attention in bf16/fp16 and to the memory-efficient kernel in fp32.
    """
    q = q.reshape(bs, heads, ch, length).transpose(2, 3).contiguous()
    k = k.reshape(bs, heads, ch, length).transpose(2, 3).contiguous()
    v = v.reshape(bs, heads, ch, length).transpose(2, 3).contiguous()
    out = F.scaled_dot_product_attention(q, k, v)            # (bs, heads, length, ch)
    return out.transpose(2, 3).reshape(bs, heads * ch, length)


def _legacy_forward(self, qkv):
    bs, width, length = qkv.shape
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
    return _sdpa(q, k, v, bs, self.n_heads, ch, length)


def _new_forward(self, qkv):
    bs, width, length = qkv.shape
    ch = width // (3 * self.n_heads)
    q, k, v = qkv.chunk(3, dim=1)
    return _sdpa(q.reshape(bs * self.n_heads, ch, length),
                 k.reshape(bs * self.n_heads, ch, length),
                 v.reshape(bs * self.n_heads, ch, length), bs, self.n_heads, ch, length)


def configure(model, enabled=True):
    """Enable or restore SDPA on the attention instances owned by ``model``."""
    for module in model.modules():
        if not isinstance(module, (_unet.QKVAttentionLegacy, _unet.QKVAttention)):
            continue
        original = getattr(module, '_neodisco_original_forward', None)
        if original is None:
            original = module.forward
            object.__setattr__(module, '_neodisco_original_forward', original)
        if not enabled:
            module.forward = original
        elif isinstance(module, _unet.QKVAttentionLegacy):
            module.forward = types.MethodType(_legacy_forward, module)
        else:
            module.forward = types.MethodType(_new_forward, module)

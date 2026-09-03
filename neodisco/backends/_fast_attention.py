"""Fused attention for the vendored guided-diffusion UNet.

The 2021 code computes attention as two einsums with an explicit softmax over the full
(tokens x tokens) matrix. At 512x512 that is fine. At 1280x768 the coarsest attention
layer sees several thousand tokens and the explicit matrix is both the slowest part of
the step and the reason fp16 overflows. `scaled_dot_product_attention` does the same
arithmetic with a fused kernel and a numerically stable softmax.

The vendored files are left untouched; this module swaps the two attention classes'
forward methods at import time. Results are identical up to floating-point rounding.
"""

import math

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


_installed = False


def install():
    """Route the UNet's attention through SDPA. Safe to call more than once."""
    global _installed
    if _installed:
        return
    _unet.QKVAttentionLegacy.forward = _legacy_forward
    _unet.QKVAttention.forward = _new_forward
    _installed = True

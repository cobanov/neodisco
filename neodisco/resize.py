"""Reuse ResizeRight's constant interpolation plans without caching image graphs."""
from collections import OrderedDict

import torch

from ._resize_right import resize_right as rr
from ._resize_right import interp_methods


class ResizeRightPlans:
    """Per-cutout-owner LRU, bounded by entries and tensor bytes across devices.

    Only geometry/weights are retained. All pixel application and backward operations
    are the original ResizeRight functions, in the original dimension order.
    """
    def __init__(self, max_entries=512, max_bytes=32 * 1024**2):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.plans = OrderedDict()
        self.bytes = 0

    def _plan(self, in_size, out_size, device):
        key = (in_size, out_size, device, torch.get_default_dtype())
        cached = self.plans.pop(key, None)
        if cached is not None:
            self.plans[key] = cached
            return cached[:3]
        scale = out_size / in_size
        # Plans must remain ordinary constant tensors even if first requested inside
        # inference_mode; a later differentiable call needs to save them for backward.
        with torch.inference_mode(False), torch.no_grad():
            eps = torch.finfo(torch.float32).eps
            grid = rr.get_projected_grid(in_size, out_size, scale, torch, False, device)
            method, support = rr.apply_antialiasing_if_needed(
                interp_methods.cubic, interp_methods.cubic.support_sz, scale, True)
            field = rr.get_field_of_view(grid, support, torch, eps, device)
            pad, grid, field = rr.calc_pad_sz(in_size, out_size, field, grid,
                                            scale, False, torch, device)
            weights = rr.get_weights(method, grid, field)
        size = field.numel()*field.element_size() + weights.numel()*weights.element_size()
        if size <= self.max_bytes and self.max_entries > 0:
            while self.plans and (len(self.plans) >= self.max_entries or
                                  self.bytes + size > self.max_bytes):
                self.bytes -= self.plans.popitem(last=False)[1][3]
            self.plans[key] = (field, weights, pad, size)
            self.bytes += size
        return field, weights, pad

    def __call__(self, pixels, size):
        dims = [(dim, size / pixels.shape[dim], pixels.shape[dim])
                for dim in (pixels.ndim-2, pixels.ndim-1) if pixels.shape[dim] != size]
        output = pixels
        for dim, _, in_size in sorted(dims, key=lambda item: item[1]):
            field, weights, pad = self._plan(in_size, size, pixels.device)
            output = rr.apply_weights(output, field, weights, dim, pixels.ndim,
                                      pad, 'constant', torch)
        return output

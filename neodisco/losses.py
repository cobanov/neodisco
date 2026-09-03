"""The loss terms Disco Diffusion used, kept faithful to the originals.

`spherical_dist_loss` is the one that matters for the look. It measures angle on the
unit sphere rather than plain cosine similarity, which keeps the gradient useful when
the image is already close to the prompt, so guidance keeps pushing detail in long runs
instead of flattening out.

`tv_loss` and `range_loss` are the guard rails: without them a long guided run drifts
out of range and turns to noise.
"""

import torch
import torch.nn.functional as F


def spherical_dist_loss(x, y):
    """Squared great-circle distance between two batches of embeddings."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(image):
    """Total variation, per image. Penalises the speckle that guidance introduces."""
    padded = F.pad(image, (0, 1, 0, 1), mode='replicate')
    x_diff = padded[..., :-1, 1:] - padded[..., :-1, :-1]
    y_diff = padded[..., 1:, :-1] - padded[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(image):
    """Penalises pixels outside [-1, 1], which guidance will happily produce."""
    return (image - image.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def saturation_loss(image):
    """Disco's later addition; holds back the neon oversaturation of long runs."""
    return (image - image.clamp(-1, 1)).abs().mean([1, 2, 3])

"""Cutouts: the reason Disco images look the way they do.

CLIP only sees square crops. Disco feeds it two kinds at once:

  overview cuts  the whole frame, squashed to CLIP's input size. These carry composition,
                 so the prompt affects the picture as a whole.
  inner cuts     random crops at random scales. These carry detail, and because a fresh
                 set is drawn every step, detail accumulates independently at every scale.

That second part is what produces the characteristic surface where every square inch is
equally busy and nothing recedes. Turning inner cuts off gives a calmer, more ordinary
image; turning them up gives the full fractal look.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class MakeCutouts(nn.Module):
    """Port of Disco Diffusion's MakeCutoutsDango, on modern torchvision."""

    def __init__(self, cut_size, overview=4, inner=32, inner_size_pow=0.5,
                 inner_grey_p=0.2, augment=True, padding_mode='constant'):
        super().__init__()
        self.cut_size = cut_size
        self.overview = overview
        self.inner = inner
        self.inner_size_pow = inner_size_pow
        self.inner_grey_p = inner_grey_p
        self.padding_mode = padding_mode
        self.grey = T.Grayscale(3)
        # Small noise between augmentations is deliberate: it stops CLIP from locking onto
        # a single high-frequency pattern and grinding it into the image.
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),
                           interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]) if augment else nn.Identity()

    def _resize(self, x):
        return F.interpolate(x, size=(self.cut_size, self.cut_size), mode='bicubic',
                             align_corners=False, antialias=True)

    def forward(self, image, overview=None, inner=None, inner_grey_p=None):
        """image: (N, 3, H, W) in [-1, 1]. Returns (n_cuts, 3, cut, cut) in [0, 1].

        Cutting and augmenting happen on the [0, 1] image, as in Disco. ColorJitter and
        the small additive noise assume that range; run them on [-1, 1] data and the
        colour augmentations misbehave, which shifts the whole run's palette.

        The three counts can be overridden per call, which is how Disco's schedules work:
        many overview cuts early for composition, many inner cuts later for detail.
        """
        overview = self.overview if overview is None else int(overview)
        inner = self.inner if inner is None else int(inner)
        inner_grey_p = self.inner_grey_p if inner_grey_p is None else inner_grey_p
        image = image.add(1).div(2)
        cuts = []
        side_y, side_x = image.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)

        if overview > 0:
            pad_y, pad_x = (side_y - max_size) // 2, (side_x - max_size) // 2
            padded = F.pad(image, (pad_x, pad_x, pad_y, pad_y), mode=self.padding_mode)
            whole = self._resize(padded)
            if overview <= 4:
                variants = [whole, self.grey(whole), TF.hflip(whole), self.grey(TF.hflip(whole))]
                cuts.extend(variants[:overview])
            else:
                cuts.extend([whole] * overview)

        grey_cutoff = int(inner_grey_p * inner)
        for i in range(inner):
            # A power law on the crop size, so small crops (fine detail) dominate.
            size = int(torch.rand([]) ** self.inner_size_pow * (max_size - min_size) + min_size)
            off_x = int(torch.randint(0, side_x - size + 1, ()))
            off_y = int(torch.randint(0, side_y - size + 1, ()))
            crop = image[:, :, off_y:off_y + size, off_x:off_x + size]
            if i <= grey_cutoff:
                crop = self.grey(crop)
            cuts.append(self._resize(crop))

        return self.augs(torch.cat(cuts))

    @property
    def n_cuts(self):
        return self.overview + self.inner

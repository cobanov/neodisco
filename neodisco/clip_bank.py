"""A bank of CLIP models, evaluated together.

Disco ran several CLIP variants at once and summed their gradients. Different backbones
disagree about what a prompt looks like, and the image ends up satisfying all of them at
once, which is part of why the results feel composited rather than designed. One model
gives a cleaner, more literal picture; three or four give the Disco look.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Name -> (open_clip model, pretrained tag). These are the ones Disco actually shipped with.
# The `-quickgelu` suffix matters: OpenAI's released weights were trained with QuickGELU,
# and open_clip warns (then silently degrades) if you load them into a plain GELU model.
DEFAULT_MODELS = [
    ('ViT-B-32-quickgelu', 'openai'),
    ('ViT-B-16-quickgelu', 'openai'),
    ('RN50-quickgelu', 'openai'),
]


class ClipBank(nn.Module):
    def __init__(self, models=None, device='cuda', dtype=torch.float32):
        super().__init__()
        import open_clip
        self.device = device
        self.models = []
        self.means = []
        self.stds = []
        self.sizes = []
        for name, pretrained in (models or DEFAULT_MODELS):
            model, _, preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, device=device)
            model = model.eval().requires_grad_(False).to(dtype)
            norm = next(t for t in preprocess.transforms if hasattr(t, 'mean'))
            self.models.append(model)
            self.means.append(torch.tensor(norm.mean, device=device).view(1, 3, 1, 1))
            self.stds.append(torch.tensor(norm.std, device=device).view(1, 3, 1, 1))
            self.sizes.append(model.visual.image_size[0]
                              if isinstance(model.visual.image_size, (tuple, list))
                              else model.visual.image_size)
            print(f'CLIP loaded: {name}/{pretrained} at {self.sizes[-1]}px')
        self.tokenizer = open_clip.get_tokenizer(models[0][0] if models else DEFAULT_MODELS[0][0])
        self.dtype = dtype

    @property
    def cut_size(self):
        """Cutouts are made at the largest input any model wants, then resized down."""
        return max(self.sizes)

    @torch.no_grad()
    def encode_text(self, prompts, weights=None):
        """Returns one (n_models, n_prompts, dim) list of embeddings plus their weights."""
        import open_clip
        weights = weights or [1.0] * len(prompts)
        out = []
        for i, model in enumerate(self.models):
            tokens = open_clip.tokenize(prompts).to(self.device)
            emb = model.encode_text(tokens).float()
            out.append(emb)
        return out, torch.tensor(weights, device=self.device, dtype=torch.float32)

    def encode_cutouts(self, cutouts, model_idx):
        """cutouts in [-1, 1]; normalises and resizes for the given model."""
        x = (cutouts + 1) / 2
        size = self.sizes[model_idx]
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bicubic',
                              align_corners=False, antialias=True)
        x = (x - self.means[model_idx]) / self.stds[model_idx]
        return self.models[model_idx].encode_image(x.to(self.dtype)).float()

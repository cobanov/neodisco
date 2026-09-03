"""CLIP guidance on a rectified-flow latent DiT (LightningDiT + VA-VAE).

The sampler is a plain Euler walk along the flow, written out by hand rather than handed
to an ODE solver, because guidance has to be injected at every step.

At time t the sample sits at x_t = t * x1 + (1 - t) * x0, and the model predicts the
velocity v = x1 - x0. So its estimate of the finished latent is x1 = x_t + (1 - t) * v.
That estimate is what gets decoded and shown to CLIP; the gradient that comes back is
subtracted from v before the step is taken.
"""

import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.environ.get('LIGHTNINGDIT_PATH',
                                  os.path.expanduser('~/Developer/latent-model/LightningDiT')))

from models.lightningdit import LightningDiT_models
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE


class LatentFlowBackend:
    name = 'latent-flow'

    def __init__(self, config_path, ckpt_path, source='ema', device='cuda'):
        self.device = device
        self.config = yaml.safe_load(open(config_path))
        cfg = self.config
        downsample = cfg.get('vae', {}).get('downsample_ratio', 16)
        self.latent_size = cfg['data']['image_size'] // downsample
        self.in_channels = cfg['model'].get('in_chans', 4)
        self.num_classes = cfg['data']['num_classes']
        self.image_size = cfg['data']['image_size']

        self.model = LightningDiT_models[cfg['model']['model_type']](
            input_size=self.latent_size,
            num_classes=self.num_classes,
            use_qknorm=cfg['model']['use_qknorm'],
            use_swiglu=cfg['model'].get('use_swiglu', False),
            use_rope=cfg['model'].get('use_rope', False),
            use_rmsnorm=cfg['model'].get('use_rmsnorm', False),
            wo_shift=cfg['model'].get('wo_shift', False),
            in_channels=self.in_channels,
        ).to(device)
        state = torch.load(ckpt_path, map_location='cpu')
        weights = state.get(source, state.get('model', state))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
        self.model.eval().requires_grad_(False)

        cfg_path = os.path.join(os.environ.get(
            'LIGHTNINGDIT_PATH', os.path.expanduser('~/Developer/latent-model/LightningDiT')),
            'tokenizer/configs', f"{cfg['vae']['model_name']}.yaml")
        self.vae = VA_VAE(cfg_path)
        self.vae.model.requires_grad_(False)

        dataset = ImgLatentDataset(cfg['data']['data_path'], latent_norm=True,
                                   latent_multiplier=cfg['data'].get('latent_multiplier', 1.0))
        mean, std = dataset.get_latent_stats()
        self.latent_mean = mean.to(device)
        self.latent_std = std.to(device)
        self.latent_multiplier = cfg['data'].get('latent_multiplier', 1.0)

    def to_pixels(self, latent, differentiable=False):
        """Normalised latent -> image tensor in [-1, 1]."""
        z = (latent * self.latent_std) / self.latent_multiplier + self.latent_mean
        if differentiable:
            return self.vae.model.decode(z)
        with torch.no_grad():
            return self.vae.model.decode(z)

    @staticmethod
    def to_uint8(pixels):
        return (torch.clamp(127.5 * pixels + 128.0, 0, 255)
                .permute(0, 2, 3, 1).to('cpu', dtype=torch.uint8).numpy())

    def sample(self, guidance=None, batch_size=1, steps=250, class_id=None, cfg_scale=1.0,
               cfg_interval_start=0.0, timestep_shift=0.0, seed=0, guidance_from=0.0,
               guidance_until=1.0, guidance_strength=1.0, cut_batch=0, progress=True):
        g = torch.Generator(device='cpu').manual_seed(seed)
        x = torch.randn(batch_size, self.in_channels, self.latent_size, self.latent_size,
                        generator=g).to(self.device)

        if class_id is None:
            y = torch.full((batch_size,), self.num_classes, device=self.device, dtype=torch.long)
            cfg_scale = 1.0  # nothing to guide toward without a class
        else:
            y = torch.full((batch_size,), int(class_id), device=self.device, dtype=torch.long)
        y_null = torch.full_like(y, self.num_classes)

        ts = torch.linspace(0, 1, steps + 1, device=self.device)
        if timestep_shift:
            s = timestep_shift
            ts = s * ts / (1 + (s - 1) * ts)

        iterator = range(steps)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc='sampling')

        for i in iterator:
            t_now, t_next = ts[i], ts[i + 1]
            dt = t_next - t_now
            t_batch = t_now.expand(batch_size)

            with torch.no_grad():
                if cfg_scale > 1.0:
                    v_cond = self.model(x, t_batch, y)
                    v_uncond = self.model(x, t_batch, y_null)
                    if t_now < cfg_interval_start:
                        v = v_cond
                    else:
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = self.model(x, t_batch, y)

            if guidance is not None and guidance_from <= float(t_now) <= guidance_until:
                # Guidance is strongest early, when there is still trajectory left to bend.
                scale = float(1.0 - t_now)
                remaining = float(1.0 - t_now)
                v_frozen = v.detach()

                def decode(x_in):
                    # The velocity is held fixed here, so the gradient flows only through
                    # x_in. Backpropagating through the transformer as well would cost a
                    # full backward pass per step for very little change in direction;
                    # Disco made the same approximation with its secondary model.
                    x1_hat = x_in + remaining * v_frozen
                    return self.to_pixels(x1_hat, differentiable=True)

                grad = guidance.gradient(x, decode, cut_batch=cut_batch)
                # Scale the gradient relative to the model's own velocity, so `strength`
                # means "guidance is this fraction as strong as the prior" and stays
                # meaningful across models, step counts and latent scales. An absolute
                # scale does not: multiplied by dt it vanishes into rounding.
                grad_rms = grad.square().mean().sqrt().clamp(min=1e-12)
                v_rms = v.square().mean().sqrt()
                v = v - grad * (guidance_strength * scale * v_rms / grad_rms)

            x = x + v * dt

        return self.to_pixels(x)

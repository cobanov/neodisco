"""The original Disco Diffusion backbones: OpenAI's unconditional ImageNet models.

Two checkpoints from 2021 and 2022, both still downloadable, both still the reason Disco
pictures look the way they do:

  256x256_diffusion_uncond.pt                    OpenAI, July 2021
  512x512_diffusion_uncond_finetune_008100.pt    Katherine Crowson's finetune, the one
                                                 Disco Diffusion used by default

Neither has ever seen a caption. They know what ImageNet textures look like and nothing
about how a picture should be arranged, which is exactly what makes CLIP guidance produce
composited, over-detailed dreams rather than tidy illustrations.

Sampling follows the original: ancestral DDPM steps with respaced timesteps, and the
guidance gradient added to the mean of each step. The clean-image estimate CLIP looks at
is recovered from the model's own noise prediction, which is what Disco's "secondary
model" was a cheap stand-in for; on a current card the real thing is affordable.
"""

import os

import numpy as np
import torch

from ..schedules import parse_schedule

from ._guided_diffusion.respace import SpacedDiffusion, space_timesteps
from ._guided_diffusion import gaussian_diffusion as gd
from ._guided_diffusion.unet import UNetModel

# Exactly the settings the released checkpoints were trained with. Getting one of these
# wrong loads silently and produces noise, so they are spelled out rather than derived.
MODEL_CONFIGS = {
    256: dict(image_size=256, model_channels=256, num_res_blocks=2,
              channel_mult=(1, 1, 2, 2, 4, 4), attention_resolutions=(8, 16, 32),
              num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True),
    512: dict(image_size=512, model_channels=256, num_res_blocks=2,
              channel_mult=(0.5, 1, 1, 2, 2, 4, 4), attention_resolutions=(16, 32, 64),
              num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True),
}

DOWNLOADS = {
    256: ('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/'
          '256x256_diffusion_uncond.pt'),
    512: ('https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/'
          'resolve/main/512x512_diffusion_uncond_finetune_008100.pt'),
}


def build_diffusion(timestep_respacing, diffusion_steps=1000, noise_schedule='linear'):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.LEARNED_RANGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=True,
    )


class PixelBackend:
    name = 'pixel'

    def __init__(self, ckpt_path, image_size=512, device='cuda', fp16=True,
                 use_checkpoint=False):
        if image_size not in MODEL_CONFIGS:
            raise ValueError(f'image_size must be 256 or 512, got {image_size}')
        self.device = device
        self.image_size = image_size
        cfg = dict(MODEL_CONFIGS[image_size])
        self.model = UNetModel(
            in_channels=3, out_channels=6, num_classes=None, dropout=0.0,
            use_fp16=fp16, use_checkpoint=use_checkpoint, num_heads=4,
            num_heads_upsample=-1, use_new_attention_order=False, **cfg)
        state = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(device).eval().requires_grad_(False)
        if fp16:
            self.model.convert_to_fp16()
        self.fp16 = fp16

    @staticmethod
    def default_path(image_size, root='weights/disco'):
        names = {256: '256x256_diffusion_uncond.pt',
                 512: '512x512_diffusion_uncond_finetune_008100.pt'}
        return os.path.join(root, names[image_size])

    @staticmethod
    def to_uint8(pixels):
        return ((pixels + 1) * 127.5).clamp(0, 255).to(torch.uint8) \
            .permute(0, 2, 3, 1).cpu().numpy()

    @torch.no_grad()
    def sample(self, guidance=None, batch_size=1, steps=250, seed=0,
               guidance_strength=1.0, cut_batch=0, eta=0.0, width=None, height=None,
               cut_overview=None, cut_innercut=None, cut_icgray_p=None, cutn_batches=1,
               clip_denoised=False, skip_steps=0, progress=True):
        """Sample an image.

        `width` and `height` may differ from the checkpoint's nominal size: the UNet is
        convolutional and its attention works on whatever grid it is given, which is how
        Disco produced 1280x768 frames from a model trained at 512x512. Both must be
        multiples of 64, since the network downsamples six times.

        `eta` selects the sampler: 0 is deterministic DDIM, 1 is ancestral DDPM, and
        Disco's usual 0.8 sits near the noisy end, which keeps injecting variety for the
        guidance to work against.
        """
        diffusion = build_diffusion(str(steps))
        g = torch.Generator(device='cpu').manual_seed(seed)
        h = height or self.image_size
        w = width or self.image_size
        if h % 64 or w % 64:
            raise ValueError(f'width and height must be multiples of 64, got {w}x{h}')
        shape = (batch_size, 3, h, w)
        x = torch.randn(*shape, generator=g).to(self.device)

        n_steps = diffusion.num_timesteps
        overview_at = parse_schedule(cut_overview, n_steps)
        inner_at = parse_schedule(cut_innercut, n_steps)
        grey_at = parse_schedule(cut_icgray_p, n_steps)

        indices = list(range(n_steps - int(skip_steps)))[::-1]
        if progress:
            from tqdm import tqdm
            indices = tqdm(indices, desc='sampling')

        alphas = diffusion.alphas_cumprod
        for i in indices:
            t = torch.tensor([i] * batch_size, device=self.device)
            out = diffusion.p_mean_variance(self.model, x, t, clip_denoised=clip_denoised)
            # Schedules are written front-to-back over the run, while i counts down.
            k = n_steps - 1 - i

            if guidance is not None:
                # CLIP looks at the model's estimate of the finished image. With the noise
                # prediction held fixed, that estimate is x scaled by 1/sqrt(alpha_bar),
                # so the gradient with respect to x is the image gradient over that same
                # factor. Reconstructing the noise term explicitly is the obvious way to
                # write this and it is unstable: near the end of sampling sqrt(1 - a_bar)
                # goes to zero and the division blows up to NaN.
                alpha_bar = float(alphas[i])
                pixel_grad = guidance.image_gradient(
                    out['pred_xstart'], cut_batch=cut_batch,
                    overview=overview_at[k] if overview_at else None,
                    inner=inner_at[k] if inner_at else None,
                    inner_grey_p=grey_at[k] if grey_at else None,
                    cutn_batches=cutn_batches)
                grad = guidance.clamp(pixel_grad / max(alpha_bar, 1e-8) ** 0.5)
                # The gradient is normalised, then scaled by the step's own noise level,
                # so `guidance_strength` means the same thing at every step and for
                # either sampler.
                grad_rms = grad.square().mean().sqrt().clamp(min=1e-12)
                step_scale = guidance_strength * out['variance'].mean() / grad_rms
                shifted = out['pred_xstart'] - grad * step_scale / max(alpha_bar, 1e-8) ** 0.5
                out['pred_xstart'] = shifted
                out['mean'] = out['mean'] - grad * step_scale

            if not torch.isfinite(out['pred_xstart']).all():
                # fp16 attention over a large frame can overflow on some seeds. Fail here
                # with a useful message rather than silently writing a blank image.
                raise FloatingPointError(
                    f'non-finite values at step {k}; rerun with fp16=False (--fp32) '
                    'or a smaller frame')

            noise = torch.randn(*shape, generator=g).to(self.device)
            if i == 0:
                x = out['pred_xstart']
                continue

            if eta <= 0:
                # Deterministic DDIM.
                ab, ab_prev = float(alphas[i]), float(alphas[i - 1])
                eps = (x - ab ** 0.5 * out['pred_xstart']) / (1 - ab) ** 0.5
                x = ab_prev ** 0.5 * out['pred_xstart'] + (1 - ab_prev) ** 0.5 * eps
            elif eta >= 1:
                x = out['mean'] + torch.exp(0.5 * out['log_variance']) * noise
            else:
                ab, ab_prev = float(alphas[i]), float(alphas[i - 1])
                eps = (x - ab ** 0.5 * out['pred_xstart']) / (1 - ab) ** 0.5
                sigma = (eta * ((1 - ab_prev) / (1 - ab)) ** 0.5
                         * (1 - ab / ab_prev) ** 0.5)
                x = (ab_prev ** 0.5 * out['pred_xstart']
                     + max(1 - ab_prev - sigma ** 2, 0.0) ** 0.5 * eps
                     + sigma * noise)

        return x




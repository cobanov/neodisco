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
from .secondary import SecondaryDiffusionImageNet2, alpha_sigma_to_t

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
                 use_checkpoint=False, secondary_path=None):
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
        # Parameters are left requiring grad on purpose. guided-diffusion's own
        # gradient checkpointing differentiates with respect to the block parameters as
        # well as the activations, and raises if they are frozen. Nothing here ever
        # accumulates into .grad, so the only cost is transient memory during backward.
        self.model.to(device).eval()
        if fp16:
            self.model.convert_to_fp16()
        self.fp16 = fp16
        # Disco's default guidance path: a small secondary model predicts the clean image
        # and the CLIP gradient is taken through it rather than through the UNet.
        self.secondary = None
        if secondary_path and os.path.exists(secondary_path):
            self.secondary = SecondaryDiffusionImageNet2()
            self.secondary.load_state_dict(torch.load(secondary_path, map_location='cpu'))
            self.secondary.to(device).eval().requires_grad_(False)

    @staticmethod
    def default_secondary_path(root='weights/disco'):
        return os.path.join(root, 'secondary_model_imagenet_2.pth')

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
               clip_denoised=False, skip_steps=0, through_model=True, disco_blend=True,
               use_secondary=True, progress=True):
        """Sample an image.

        `width` and `height` may differ from the checkpoint's nominal size: the UNet is
        convolutional and its attention works on whatever grid it is given, which is how
        Disco produced 1280x768 frames from a model trained at 512x512. Both must be
        multiples of 64, since the network downsamples six times.

        `eta` selects the sampler: 0 is deterministic DDIM, 1 is ancestral DDPM, and
        Disco's usual 0.8 sits near the noisy end, which keeps injecting variety for the
        guidance to work against.
        """
        # Disco respaces as 'ddim<N>' (every 1000//N-th timestep from 0), which is not the
        # same set of timesteps as the plain '<N>' spacing.
        # 'ddim<N>' needs N to divide 1000 (250, 200, 125, 100, 50...). For any other
        # count fall back to the plain even spacing rather than refusing to run.
        spacing = f'ddim{steps}' if 1000 % int(steps) == 0 else str(steps)
        diffusion = build_diffusion(spacing)
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
                # This follows Disco's cond_fn. Three details matter for the look and are
                # easy to get wrong:
                #
                # 1. CLIP is shown a blend, x_in = x0_hat * s + x * (1 - s) with
                #    s = sqrt(1 - alpha_bar), not the clean estimate alone. Early on that
                #    is the estimate; late in the run it is mostly the sample itself.
                # 2. The gradient goes back through the UNet (Disco used a small
                #    "secondary" model as a cheap stand-in; here the real one is used).
                # 3. The gradient is clamped to an RMS of clamp_max, and that clamped
                #    value is what the sampler scales by the step's noise level. Rescaling
                #    it to any other magnitude blows the colours out within a few dozen
                #    steps; the first version of this file did exactly that.
                ab = float(alphas[i])
                s_t = (1 - ab) ** 0.5
                if self.secondary is not None and use_secondary:
                    # x_in = secondary.pred * s + x * (1 - s), gradient through the
                    # secondary model. This is Disco's cond_fn with use_secondary_model on.
                    with torch.enable_grad():
                        x_g = x.detach().requires_grad_(True)
                        alpha = torch.tensor(ab ** 0.5, device=self.device)
                        sigma = torch.tensor(s_t, device=self.device)
                        cosine_t = alpha_sigma_to_t(alpha, sigma)
                        pred = self.secondary(x_g, cosine_t[None].repeat(x_g.shape[0])).pred
                        x_in = pred * s_t + x_g * (1 - s_t)
                        pixel_grad = guidance.image_gradient(
                            x_in, cut_batch=cut_batch,
                            overview=overview_at[k] if overview_at else None,
                            inner=inner_at[k] if inner_at else None,
                            inner_grey_p=grey_at[k] if grey_at else None,
                            cutn_batches=cutn_batches)
                        grad = torch.autograd.grad(x_in, x_g, grad_outputs=pixel_grad)[0]
                elif through_model:
                    with torch.enable_grad():
                        x_g = x.detach().requires_grad_(True)
                        out_g = diffusion.p_mean_variance(self.model, x_g, t,
                                                          clip_denoised=clip_denoised)
                        x_in = (out_g['pred_xstart'] * s_t + x_g * (1 - s_t)
                                if disco_blend else out_g['pred_xstart'])
                        pixel_grad = guidance.image_gradient(
                            x_in, cut_batch=cut_batch,
                            overview=overview_at[k] if overview_at else None,
                            inner=inner_at[k] if inner_at else None,
                            inner_grey_p=grey_at[k] if grey_at else None,
                            cutn_batches=cutn_batches)
                        grad = torch.autograd.grad(x_in, x_g, grad_outputs=pixel_grad)[0]
                else:
                    x_in = (out['pred_xstart'] * s_t + x * (1 - s_t)
                            if disco_blend else out['pred_xstart'])
                    pixel_grad = guidance.image_gradient(
                        x_in, cut_batch=cut_batch,
                        overview=overview_at[k] if overview_at else None,
                        inner=inner_at[k] if inner_at else None,
                        inner_grey_p=grey_at[k] if grey_at else None,
                        cutn_batches=cutn_batches)
                    # Jacobian of x_in with the noise prediction held fixed.
                    grad = pixel_grad * ((1 - s_t) + s_t / max(ab, 1e-3) ** 0.5)

                if guidance.clamp_max:
                    cond = -guidance.clamp(grad)
                else:
                    # No clamp: fall back to a step-relative scale.
                    grad_rms = grad.square().mean().sqrt().clamp(min=1e-12)
                    cond = -grad * (guidance_strength * out['variance'].mean() / grad_rms)

                if eta < 1:
                    # guided-diffusion's condition_score: guidance enters through eps.
                    eps = (x - ab ** 0.5 * out['pred_xstart']) / (1 - ab) ** 0.5
                    eps = eps - (1 - ab) ** 0.5 * cond
                    out['pred_xstart'] = (x - (1 - ab) ** 0.5 * eps) / ab ** 0.5
                else:
                    # condition_mean: guidance moves the posterior mean.
                    out['mean'] = out['mean'] + out['variance'] * cond

            if not torch.isfinite(out['pred_xstart']).all():
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




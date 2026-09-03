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
from . import _fast_attention

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
        rescale_timesteps=False,
    )


class PixelBackend:
    name = 'pixel'

    def __init__(self, ckpt_path, image_size=512, device='cuda', fp16=True,
                 use_checkpoint=False, secondary_path=None, fast_attention=True,
                 autocast_dtype=None):
        if image_size not in MODEL_CONFIGS:
            raise ValueError(f'image_size must be 256 or 512, got {image_size}')
        if fast_attention:
            _fast_attention.install()
        # bf16 autocast for the UNet and secondary model forward passes. Unlike fp16
        # weights it does not overflow on wide frames, and on Ampere and newer it is
        # close to fp16 speed. None keeps everything in fp32.
        self.autocast_dtype = autocast_dtype
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

    def _autocast(self):
        import contextlib
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast('cuda', dtype=self.autocast_dtype)

    @staticmethod
    def default_secondary_path(root='weights/disco'):
        return os.path.join(root, 'secondary_model_imagenet_2.pth')

    @staticmethod
    def default_path(image_size, root='weights/disco'):
        names = {256: '256x256_diffusion_uncond.pt',
                 512: '512x512_diffusion_uncond_finetune_008100.pt'}
        return os.path.join(root, names[image_size])

    @staticmethod
    def _load_init(image, w, h):
        """Path, PIL image or tensor -> (1, 3, h, w) in [-1, 1], resized to the frame."""
        from PIL import Image
        if torch.is_tensor(image):
            t = image.float()
            if t.dim() == 3:
                t = t[None]
        else:
            pil = Image.open(image) if isinstance(image, str) else image
            pil = pil.convert('RGB').resize((w, h), Image.LANCZOS)
            t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1)[None].float() / 127.5 - 1
        if t.shape[-2:] != (h, w):
            t = torch.nn.functional.interpolate(t, size=(h, w), mode='bicubic',
                                                align_corners=False, antialias=True)
        return t

    @staticmethod
    def to_uint8(pixels):
        return ((pixels + 1) * 127.5).clamp(0, 255).to(torch.uint8) \
            .permute(0, 2, 3, 1).cpu().numpy()

    @torch.no_grad()
    def sample(self, guidance=None, batch_size=1, steps=250, seed=0,
               guidance_strength=1.0, cut_batch=0, eta=0.0, width=None, height=None,
               cut_overview=None, cut_innercut=None, cut_icgray_p=None, cutn_batches=1,
               clip_denoised=False, skip_steps=0, through_model=True, disco_blend=True,
               use_secondary=True, init_image=None, init_scale=0.0, progress=True):
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

        # Disco's init image: start from the image noised to the first step actually run,
        # rather than from pure noise. With skip_steps at about half the run, the
        # composition of the init survives and the guidance repaints the detail; this
        # is also how Disco upscaled, rendering small and re-running large from the result.
        init = None
        if init_image is not None:
            init = self._load_init(init_image, w, h).to(self.device).expand(batch_size, -1, -1, -1)
            if guidance is not None:
                guidance.set_init(init, init_scale)

        n_steps = diffusion.num_timesteps
        overview_at = parse_schedule(cut_overview, n_steps)
        inner_at = parse_schedule(cut_innercut, n_steps)
        grey_at = parse_schedule(cut_icgray_p, n_steps)
        alphas = diffusion.alphas_cumprod

        indices = list(range(n_steps - int(skip_steps)))[::-1]
        if init is not None:
            t_start = torch.tensor([indices[0]] * batch_size, device=self.device)
            x = diffusion.q_sample(init, t_start, noise=x)

        # Guidance as a cond_fn, exactly the shape guided-diffusion's ddim_sample wants:
        # cond_fn(x, t) returns grad(log p(prompt | x)), the descent direction on the CLIP
        # loss, clamped. The vendored ddim_sample then folds it into eps via condition_score
        # and takes the step with the reference DDIM math. Rolling that step by hand was
        # what drifted the tonality: alphas_cumprod_prev on the respaced schedule is not
        # alphas_cumprod[i-1], and the mismatch quietly darkened and over-saturated
        # every run.
        def make_cond_fn(i):
            # The step index is bound here rather than read from the tensor: guided-diffusion
            # hands cond_fn the timestep after _scale_timesteps, which on a respaced
            # schedule is the original 1000-space value, not the index into our arrays.
            def cond_fn(x_t, _t):
                return _guidance_grad(x_t, i)
            return cond_fn

        def _guidance_grad(x_t, i):
            if guidance is None:
                return torch.zeros_like(x_t)
            k = n_steps - 1 - i
            ab = float(alphas[i])
            s_t = (1 - ab) ** 0.5
            with torch.enable_grad():
                x_g = x_t.detach().requires_grad_(True)
                if self.secondary is not None and use_secondary:
                    cosine_t = alpha_sigma_to_t(torch.tensor(ab ** 0.5, device=self.device),
                                                torch.tensor(s_t, device=self.device))
                    with self._autocast():
                        pred = self.secondary(x_g, cosine_t[None].repeat(x_g.shape[0])).pred
                    pred = pred.float()
                else:
                    with self._autocast():
                        og = diffusion.p_mean_variance(self.model, x_g,
                                                       torch.tensor([i] * x_g.shape[0], device=self.device),
                                                       clip_denoised=clip_denoised)
                    pred = og['pred_xstart'].float()
                x_in = pred * s_t + x_g * (1 - s_t) if disco_blend else pred
                pixel_grad = guidance.image_gradient(
                    x_in, cut_batch=cut_batch,
                    overview=overview_at[k] if overview_at else None,
                    inner=inner_at[k] if inner_at else None,
                    inner_grey_p=grey_at[k] if grey_at else None,
                    cutn_batches=cutn_batches,
                    range_target=(lambda probe, _x=x_g.detach(), _s=s_t:
                                  (probe - _x * (1 - _s)) / max(_s, 1e-3)))
                grad = torch.autograd.grad(x_in, x_g, grad_outputs=pixel_grad)[0]
            # Disco returns -grad(loss), clamped to clamp_max, as the score direction.
            return -guidance.clamp(grad)

        iterator = indices
        if progress:
            from tqdm import tqdm
            iterator = tqdm(indices, desc='sampling')

        for i in iterator:
            t = torch.tensor([i] * batch_size, device=self.device)
            with torch.no_grad(), self._autocast():
                out = diffusion.ddim_sample(self.model, x, t, clip_denoised=clip_denoised,
                                            cond_fn=make_cond_fn(i) if guidance is not None else None,
                                            eta=eta, model_kwargs={})
            x = out['sample'].float()

        return x




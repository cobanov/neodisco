"""The original Disco Diffusion backbones: OpenAI's unconditional ImageNet models.

Two checkpoints from 2021 and 2022, both still downloadable, both still the reason Disco
pictures look the way they do:

  256x256_diffusion_uncond.pt                    OpenAI, July 2021
  512x512_diffusion_uncond_finetune_008100.pt    Katherine Crowson's finetune, the one
                                                 Disco Diffusion used by default

Neither has ever seen a caption. They know what ImageNet textures look like and nothing
about how a picture should be arranged, which is exactly what makes CLIP guidance produce
composited, over-detailed dreams rather than tidy illustrations.

Sampling uses the vendored reference DDIM score-conditioning update on Disco's respaced
timesteps. The clean-image estimate CLIP looks at comes from the secondary model by
default, or through the original UNet when secondary guidance is disabled.
"""

import os
import warnings

import numpy as np
import torch
from torch.profiler import record_function

from ..schedules import parse_schedule
from ..runtime import autocast_context, render_rng, fp32_context

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
        rescale_timesteps=True,
    )


class PixelBackend:
    name = 'pixel'

    def __init__(self, ckpt_path, image_size=512, device='cuda', fp16=False,
                 use_checkpoint=False, secondary_path=None, fast_attention=True,
                 autocast_dtype=None, compile_mode='eager', compiler=None):
        if image_size not in MODEL_CONFIGS:
            raise ValueError(f'image_size must be 256 or 512, got {image_size}')
        if secondary_path and not os.path.exists(secondary_path):
            raise FileNotFoundError(f'secondary model not found: {secondary_path}')
        # bf16 autocast for UNet forwards. Secondary guidance stays fp32 as in Disco. Unlike fp16
        # weights it does not overflow on wide frames, and on Ampere and newer it is
        # close to fp16 speed. None keeps everything in fp32.
        self.autocast_dtype = autocast_dtype
        self.device = torch.device(device)
        self.image_size = image_size
        cfg = dict(MODEL_CONFIGS[image_size])
        self.model = UNetModel(
            in_channels=3, out_channels=6, num_classes=None, dropout=0.0,
            use_fp16=fp16, use_checkpoint=use_checkpoint, num_heads=4,
            num_heads_upsample=-1, use_new_attention_order=False, **cfg)
        state = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state)
        # Guidance differentiates with respect to the noisy image, never checkpoint
        # weights. The maintained non-reentrant checkpoint path supports frozen weights.
        self.model.to(self.device).eval().requires_grad_(False)
        if fp16:
            self.model.convert_to_fp16()
        _fast_attention.configure(self.model, enabled=fast_attention)
        self.attention_mode = 'sdpa' if fast_attention else 'original'
        self.fp16 = fp16
        self.compile_mode = compile_mode
        self.effective_compile_mode = compile_mode
        self._compiled_model = self.model
        if compile_mode != 'eager':
            compile_fn = compiler or torch.compile
            try:
                self._compiled_model = compile_fn(self.model, mode=compile_mode, fullgraph=False)
            except Exception as exc:
                if self._contains_fatal_failure(exc):
                    raise
                warnings.warn(f'torch.compile setup failed; using eager UNet: {exc}', RuntimeWarning)
                self._compiled_model = self.model
                self.effective_compile_mode = 'eager-fallback'
        # Disco's default guidance path: a small secondary model predicts the clean image
        # and the CLIP gradient is taken through it rather than through the UNet.
        self.secondary = None
        if secondary_path:
            self.secondary = SecondaryDiffusionImageNet2()
            self.secondary.load_state_dict(torch.load(secondary_path, map_location='cpu'))
            self.secondary.to(self.device).eval().requires_grad_(False)

    def _autocast(self):
        return autocast_context(self.device, self.autocast_dtype)

    @staticmethod
    def _contains_fatal_failure(exc):
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if isinstance(exc, (torch.OutOfMemoryError, FloatingPointError)):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    @classmethod
    def _is_compile_failure(cls, exc):
        if cls._contains_fatal_failure(exc):
            return False
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if type(exc).__module__.startswith(('torch._dynamo', 'torch._inductor')):
                return True
            exc = exc.__cause__ or exc.__context__
        return False

    def _model_forward(self, *args, **kwargs):
        try:
            with record_function('neodisco.unet'), self._autocast():
                return self._compiled_model(*args, **kwargs)
        except Exception as exc:
            if self._compiled_model is self.model or not self._is_compile_failure(exc):
                raise
            warnings.warn(f'torch.compile execution failed; using eager UNet: {exc}', RuntimeWarning)
            self._compiled_model = self.model
            self.effective_compile_mode = 'eager-fallback'
            with record_function('neodisco.unet'), self._autocast():
                return self.model(*args, **kwargs)

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
               guidance_strength=None, cut_batch=0, eta=0.0, width=None, height=None,
               cut_overview=None, cut_innercut=None, cut_icgray_p=None, cutn_batches=1,
               clip_denoised=False, skip_steps=0, through_model=True, disco_blend=True,
               use_secondary=True, init_image=None, init_scale=0.0, progress=True,
               preview=None, deterministic=False, finite_check=True, cut_ic_pow=None):
        """Sample an image.

        `width` and `height` may differ from the checkpoint's nominal size: the UNet is
        convolutional and its attention works on whatever grid it is given, which is how
        Disco produced 1280x768 frames from a model trained at 512x512. Both must be
        multiples of 64, since the network downsamples six times.

        `eta=0` makes the DDIM update deterministic; positive eta injects stochastic
        DDIM noise, with Disco's usual 0.8 near the noisy end. Full render repetition also
        requires deterministic cutout/augmentation kernels, enabled by reference mode.
        """
        if guidance_strength is not None:
            raise ValueError(
                'guidance_strength/--strength is retired: its earlier sampler-relative '
                'formula was not valid for the reference DDIM update; use clamp_max'
            )
        if through_model is not True:
            raise ValueError(
                'through_model=False is no longer supported by the pixel backend; '
                'use the secondary model or the differentiable UNet path'
            )
        if int(batch_size) < 1:
            raise ValueError('batch_size must be positive')
        if init_scale and init_image is None:
            raise ValueError('init_scale requires init_image')
        if init_scale and guidance is None:
            raise ValueError('init_scale requires a guidance object')
        steps, skip_steps = int(steps), int(skip_steps)
        if not 1 <= steps <= 1000:
            raise ValueError('steps must be between 1 and 1000')
        if not 0 <= skip_steps < steps:
            raise ValueError('skip_steps must be at least 0 and less than steps')
        if not 0 <= float(eta) <= 1:
            raise ValueError('eta must be between 0 and 1')
        # Disco adjusts the base process so every requested count has a DDIM stride.
        base_steps = (1000 // steps) * steps
        diffusion = build_diffusion(f'ddim{steps}', diffusion_steps=base_steps)
        scaled_times = (torch.tensor(diffusion.timestep_map).float()
                        * (1000.0 / base_steps)).tolist()
        schedule_indices = [999 - int(t) for t in scaled_times]
        h = height or self.image_size
        w = width or self.image_size
        if h <= 0 or w <= 0 or h % 64 or w % 64:
            raise ValueError(f'width and height must be positive multiples of 64, got {w}x{h}')
        shape = (batch_size, 3, h, w)
        self.guidance_nan_steps = []
        with render_rng(seed, self.device, deterministic=deterministic):
            x = torch.randn(shape, device=self.device)
            if guidance is not None:
                guidance.set_init(None, 0.0)

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
            overview_at = parse_schedule(cut_overview, 1000)
            inner_at = parse_schedule(cut_innercut, 1000)
            grey_at = parse_schedule(cut_icgray_p, 1000)
            power_at = parse_schedule(cut_ic_pow, 1000)
            alphas = diffusion.alphas_cumprod

            indices = list(range(n_steps - skip_steps))[::-1]
            if init is not None or skip_steps:
                # The original fork noises a zero init when skipping without an image.
                noise_init = init if init is not None else torch.zeros_like(x)
                t_start = torch.tensor([indices[0]] * batch_size, device=self.device)
                x = diffusion.q_sample(noise_init, t_start, noise=x)

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
                    with record_function('neodisco.guidance_total'):
                        return _guidance_grad(x_t, i)
                return cond_fn

            def _guidance_grad(x_t, i):
                if guidance is None:
                    return torch.zeros_like(x_t)
                k = schedule_indices[i]
                ab = float(alphas[i])
                s_t = (1 - ab) ** 0.5
                with torch.enable_grad():
                    x_g = x_t.detach().float().requires_grad_(True)
                    if self.secondary is not None and use_secondary:
                        cosine_t = alpha_sigma_to_t(torch.tensor(ab ** 0.5, device=self.device),
                                                    torch.tensor(s_t, device=self.device))
                        with record_function('neodisco.secondary'), fp32_context(self.device):
                            pred = self.secondary(x_g, cosine_t[None].repeat(x_g.shape[0])).pred
                        pred = pred.float()
                    else:
                        og = diffusion.p_mean_variance(
                            self._model_forward, x_g,
                            torch.tensor([i] * x_g.shape[0], device=self.device),
                            clip_denoised=False)
                        pred = og['pred_xstart'].float()
                    if not torch.isfinite(pred).all().item():
                        raise FloatingPointError(f'non-finite guidance prediction at timestep {i}')
                    x_in = pred * s_t + x_g * (1 - s_t) if disco_blend else pred
                    pixel_grad = guidance.image_gradient(
                        x_in, cut_batch=cut_batch,
                        overview=overview_at[k] if overview_at else None,
                        inner=inner_at[k] if inner_at else None,
                        inner_grey_p=grey_at[k] if grey_at else None,
                        cutn_batches=cutn_batches,
                        deterministic=deterministic, allow_nan=disco_blend,
                        inner_size_pow=power_at[k] if power_at else None,
                        # pred is upstream of x_in in Disco, so range(pred) has zero
                        # derivative with respect to x_in. Never invert the blend.
                        range_target=(lambda probe: pred.detach()) if disco_blend else None)
                    if disco_blend and torch.isnan(pixel_grad).any().item():
                        # Original cond_fn skips only the invalid CLIP/image guidance,
                        # then continues the otherwise finite diffusion step.
                        if not self.guidance_nan_steps:
                            warnings.warn('NaN image guidance: skipping affected steps as in '
                                          'original Disco; recorded in guidance_nan_steps',
                                          RuntimeWarning)
                        self.guidance_nan_steps.append(len(indices) - i)
                        return torch.zeros_like(x_t)
                    with record_function('neodisco.guidance_backbone_backward'):
                        grad = torch.autograd.grad(x_in, x_g, grad_outputs=pixel_grad)[0]
                if finite_check and not torch.isfinite(grad).all().item():
                    raise FloatingPointError(
                        f'non-finite guidance gradient at step {k + 1}/{len(indices)} '
                        f'(precision={self.autocast_dtype}, attention={self.attention_mode})'
                    )
                # Disco returns -grad(loss), clamped to clamp_max, as the score direction.
                return -guidance.clamp(grad)

        # `progress` is either a flag (show a tqdm bar) or a callable that wraps the
        # iterator, which is how the server reports step counts to the browser.
            if callable(progress):
                iterator = progress(indices)
            elif progress:
                from tqdm import tqdm
                iterator = tqdm(indices, desc='sampling')
            else:
                iterator = indices

        # pred_xstart is the model's guess at the finished image from where it currently
        # stands, which is what Disco showed while a render was running: recognisable
        # early, sharp late. It costs nothing here, the step already computed it.
            for n, i in enumerate(iterator, start=1):
                t = torch.tensor([i] * batch_size, device=self.device)
                with torch.no_grad():
                    out = diffusion.ddim_sample(
                        self._model_forward, x, t, clip_denoised=clip_denoised,
                        cond_fn=make_cond_fn(i) if guidance is not None else None,
                        eta=eta, model_kwargs={})
                x = out['sample'].float()
                if finite_check:
                    for name, value in (('sample', x), ('pred_xstart', out['pred_xstart']),
                                        ('pred_xstart_uncond', out['pred_xstart_uncond'])):
                        if not torch.isfinite(value).all().item():
                            raise FloatingPointError(
                                f'non-finite {name} at step {n}/{len(indices)} '
                                f'(seed={seed}, size={w}x{h}, precision={self.autocast_dtype}, '
                                f'attention={self.attention_mode})'
                            )
                if preview is not None:
                    preview(n, out['pred_xstart_uncond'].float())

            return out['pred_xstart_uncond'].float()

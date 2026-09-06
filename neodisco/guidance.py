"""The guidance term: how far the current image is from the prompt, and which way to move.

Every sampling step, the model's estimate of the finished image is decoded to pixels, cut
into crops, embedded by each CLIP, and compared to the prompt embeddings. The gradient of
that distance with respect to the noisy sample is subtracted from the model's own
prediction. The model pulls toward what it was trained on; CLIP pulls toward the prompt;
the picture is whatever survives both.
"""

import torch
from torch.profiler import record_function

from .losses import spherical_dist_loss, tv_loss, range_loss, saturation_loss
from .runtime import fp32_context


class PromptGuidance:
    def __init__(self, clip_bank, cutouts, prompts, weights=None,
                 clip_scale=5000.0, tv_scale=0.0, range_scale=150.0, sat_scale=0.0,
                 clamp_max=0.0):
        self.bank = clip_bank
        self.cutouts = cutouts
        self.embeddings, self.weights = clip_bank.encode_text(prompts, weights)
        self.clip_scale = clip_scale
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.clamp_max = clamp_max
        self.init = None
        self.init_scale = 0.0
        self._lpips = None

    def set_init(self, init, init_scale, perceptual_model=None):
        """Keep the sample perceptually close to an init image (Disco's init_scale).

        Uses LPIPS with the VGG backbone in the same [-1, 1] image domain as Disco.
        A nonzero scale is never silently ignored.
        """
        self.init = init
        self.init_scale = float(init_scale)
        if self.init is None or not self.init_scale:
            return
        if perceptual_model is not None:
            self._lpips = perceptual_model.to(init.device).eval().requires_grad_(False)
            return
        if self._lpips is not None:
            return
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                'init_scale requires LPIPS; install neodisco[init] or set init_scale=0'
            ) from exc
        self._lpips = lpips.LPIPS(net='vgg', verbose=False).to(init.device).eval()
        self._lpips.requires_grad_(False)

    def _perceptual_term(self, pixels):
        if not self.init_scale:
            return pixels.new_zeros(())
        if self._lpips is None or self.init is None:
            raise RuntimeError('init_scale is nonzero but LPIPS guidance is unavailable')
        init = self.init.to(device=pixels.device, dtype=torch.float32)
        if init.shape[0] == 1 and pixels.shape[0] != 1:
            init = init.expand(pixels.shape[0], -1, -1, -1)
        return self._lpips(pixels.float(), init).sum() * self.init_scale

    def _clip_term(self, cuts, deterministic=False):
        """Mean spherical distance between a set of cutouts and the prompts."""
        total = torch.zeros((), device=self.bank.device, dtype=torch.float32)
        for i in range(len(self.bank.models)):
            emb = self.bank.encode_cutouts(cuts, i, deterministic=deterministic)
            # (cuts, 1, d) against (1, prompts, d) -> (cuts, prompts)
            dists = spherical_dist_loss(emb.unsqueeze(1), self.embeddings[i].unsqueeze(0))
            total = total + (dists * self.weights).sum(dim=1).mean()
        # Disco sums over CLIP models rather than averaging; more models pull harder.
        return total

    def loss(self, pixels):
        """pixels: (N, 3, H, W) in [-1, 1], part of a live autograd graph."""
        with fp32_context(pixels.device):
            pixels = pixels.float()
            cuts = self.cutouts(pixels)
            total = pixels.new_zeros(())
            for i in range(len(self.bank.models)):
                emb = self.bank.encode_cutouts(cuts, i)
                dists = spherical_dist_loss(emb.unsqueeze(1), self.embeddings[i].unsqueeze(0))
                total = total + (dists * self.weights).sum(dim=1).mean()
            out = total * self.clip_scale + self._perceptual_term(pixels)
            if self.tv_scale:
                out = out + tv_loss(pixels).sum() * self.tv_scale
            if self.range_scale:
                out = out + range_loss(pixels).sum() * self.range_scale
            if self.sat_scale:
                out = out + saturation_loss(pixels).sum() * self.sat_scale
            return out

    def image_gradient(self, pixels, cut_batch=0, overview=None, inner=None,
                       inner_grey_p=None, cutn_batches=1, range_target=None,
                       deterministic=False):
        """d(loss) / d(pixels), accumulated over groups of cutouts.

        The cutouts and CLIP are where the activation memory goes, so they are done in
        groups of `cut_batch` and their gradients summed. `pixels` is treated as a leaf:
        callers that produced it from something else push this gradient the rest of the
        way themselves.
        """
        # Disco redraws the cutouts several times per step and averages the gradients.
        # One draw is a noisy estimate of "what the prompt wants here"; averaging a few
        # steadies it without changing what it asks for. All draws are made up front and
        # scored in one pass: the mean over the union equals the mean of per-draw means,
        # and the GPU sees a few large CLIP batches instead of many small ones.
        return self._one_draw(pixels, cut_batch, overview, inner, inner_grey_p,
                              draws=max(int(cutn_batches), 1), range_target=range_target,
                              deterministic=deterministic)

    def _one_draw(self, pixels, cut_batch, overview, inner, inner_grey_p, draws=1,
                  range_target=None, deterministic=False):
        with fp32_context(pixels.device), torch.enable_grad():
            probe = pixels.detach().float().requires_grad_(True)
            # CUDA grid_sample backward is nondeterministic in the torchvision affine
            # augmentation. Reference mode keeps the same transformations and random
            # draws on CPU, then copies cutouts to the CLIP device through autograd.
            cut_source = probe.cpu() if deterministic and probe.device.type == 'cuda' else probe
            with record_function('neodisco.cutouts'):
                cut_sets = [self.cutouts(cut_source, overview=overview, inner=inner,
                                         inner_grey_p=inner_grey_p) for _ in range(draws)]
            if not cut_sets or not cut_sets[0].shape[0]:
                raise ValueError('guidance requires at least one cutout')
            cuts = torch.cat(cut_sets)
            n = cuts.shape[0]
            size = cut_batch if cut_batch and cut_batch < n else n
            starts = list(range(0, n, size))
            grad = torch.zeros_like(probe)

            for k, begin in enumerate(starts):
                chunk = cuts[begin:begin + size]
                with record_function('neodisco.clip_guidance'):
                    term = self._clip_term(chunk, deterministic=deterministic) * (chunk.shape[0] / n) * self.clip_scale
                if k == len(starts) - 1:
                    # These depend on the whole image rather than any one cutout, so they
                    # ride along with the final group.
                    if self.tv_scale:
                        term = term + tv_loss(probe).sum() * self.tv_scale
                    if self.range_scale:
                        # Disco measures the range penalty on the clean prediction (the
                        # secondary model's output), not on the blend CLIP looks at.
                        # `range_target` carries that prediction as a function of probe.
                        target = range_target(probe) if range_target is not None else probe
                        term = term + range_loss(target).sum() * self.range_scale
                    if self.sat_scale:
                        term = term + saturation_loss(probe).sum() * self.sat_scale
                    # Disco applies LPIPS to the blended x_in image in [-1, 1]. It is an
                    # image loss, so it is added once rather than once per CLIP chunk or
                    # cutout draw.
                    term = term + self._perceptual_term(probe)
                with record_function('neodisco.guidance_loss_backward'):
                    grad = grad + torch.autograd.grad(
                        term, probe, retain_graph=(k < len(starts) - 1))[0]
        self.ensure_finite(grad, 'guidance image gradient')
        return grad

    def gradient(self, x, decode_fn, cut_batch=0, **cut_kwargs):
        """Gradient of the guidance loss with respect to the sample `x`.

        `decode_fn` turns the model's clean-image estimate into pixels and must stay
        differentiable, so for a latent model it runs the VAE decoder rather than the
        no-grad helper. The image-space gradient is found first, in groups, and pushed
        back through the decoder exactly once; a decoder backward per group would be
        several times slower for the same answer.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            pixels = decode_fn(x)
            pixel_grad = self.image_gradient(pixels, cut_batch=cut_batch, **cut_kwargs)
            grad = torch.autograd.grad(pixels, x, grad_outputs=pixel_grad)[0]
        return self.clamp(grad)

    def clamp(self, grad):
        """Bound the step size. Without this a single confident step wrecks the image."""
        self.ensure_finite(grad, 'guidance gradient')
        if not self.clamp_max:
            return grad
        dims = list(range(1, grad.dim()))
        peak = grad.abs().amax(dim=dims, keepdim=True)
        safe = peak.clamp(min=torch.finfo(grad.dtype).tiny)
        magnitude = safe * (grad.div(safe).square().mean(dim=dims, keepdim=True).sqrt())
        out = grad * magnitude.clamp(max=self.clamp_max) / magnitude.clamp(min=1e-12)
        self.ensure_finite(out, 'clamped guidance score')
        return out

    @staticmethod
    def ensure_finite(value, context):
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f'non-finite values in {context}')

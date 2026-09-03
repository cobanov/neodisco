"""The guidance term: how far the current image is from the prompt, and which way to move.

Every sampling step, the model's estimate of the finished image is decoded to pixels, cut
into crops, embedded by each CLIP, and compared to the prompt embeddings. The gradient of
that distance with respect to the noisy sample is subtracted from the model's own
prediction. The model pulls toward what it was trained on; CLIP pulls toward the prompt;
the picture is whatever survives both.
"""

import torch

from .losses import spherical_dist_loss, tv_loss, range_loss, saturation_loss


class PromptGuidance:
    def __init__(self, clip_bank, cutouts, prompts, weights=None,
                 clip_scale=1.0, tv_scale=0.0, range_scale=0.0, sat_scale=0.0,
                 clamp_max=0.0):
        self.bank = clip_bank
        self.cutouts = cutouts
        self.embeddings, self.weights = clip_bank.encode_text(prompts, weights)
        self.clip_scale = clip_scale
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.clamp_max = clamp_max

    def _clip_term(self, cuts):
        """Mean spherical distance between a set of cutouts and the prompts."""
        total = cuts.new_zeros(())
        for i in range(len(self.bank.models)):
            emb = self.bank.encode_cutouts(cuts, i)
            # (cuts, 1, d) against (1, prompts, d) -> (cuts, prompts)
            dists = spherical_dist_loss(emb.unsqueeze(1), self.embeddings[i].unsqueeze(0))
            total = total + (dists * self.weights).sum(dim=1).mean()
        return total / max(len(self.bank.models), 1)

    def loss(self, pixels):
        """pixels: (N, 3, H, W) in [-1, 1], part of a live autograd graph."""
        cuts = self.cutouts(pixels)
        n_cuts = self.cutouts.n_cuts
        total = pixels.new_zeros(())
        for i in range(len(self.bank.models)):
            emb = self.bank.encode_cutouts(cuts, i)
            # (cuts, 1, d) against (1, prompts, d) -> (cuts, prompts)
            dists = spherical_dist_loss(emb.unsqueeze(1), self.embeddings[i].unsqueeze(0))
            total = total + (dists * self.weights).sum(dim=1).mean()
        total = total / max(len(self.bank.models), 1)

        out = total * self.clip_scale
        if self.tv_scale:
            out = out + tv_loss(pixels).sum() * self.tv_scale
        if self.range_scale:
            out = out + range_loss(pixels).sum() * self.range_scale
        if self.sat_scale:
            out = out + saturation_loss(pixels).sum() * self.sat_scale
        return out

    def gradient(self, x, decode_fn, cut_batch=0):
        """Gradient of the guidance loss with respect to the sample `x`.

        `decode_fn` turns the model's clean-image estimate into pixels and must stay
        differentiable, so for a latent model it runs the VAE decoder rather than the
        no-grad helper.

        The work is split in two so that a large cut count stays affordable. The cutouts
        and CLIP are differentiated with respect to the *image*, in groups of `cut_batch`,
        which is where the activation memory goes. Those partial image gradients are summed
        and pushed through the decoder exactly once. Doing it the naive way runs a decoder
        backward per group and is several times slower for the same answer.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            pixels = decode_fn(x)

            # Stage one: d(loss) / d(pixels), accumulated over cutout groups.
            probe = pixels.detach().requires_grad_(True)
            cuts = self.cutouts(probe)
            n = cuts.shape[0]
            size = cut_batch if cut_batch and cut_batch < n else n
            starts = list(range(0, n, size))
            pixel_grad = torch.zeros_like(probe)

            for k, begin in enumerate(starts):
                chunk = cuts[begin:begin + size]
                term = self._clip_term(chunk) * (chunk.shape[0] / n) * self.clip_scale
                if k == len(starts) - 1:
                    # The pixel-space terms depend on the whole image, not on any one
                    # cutout, so they ride along with the final group.
                    if self.tv_scale:
                        term = term + tv_loss(probe).sum() * self.tv_scale
                    if self.range_scale:
                        term = term + range_loss(probe).sum() * self.range_scale
                    if self.sat_scale:
                        term = term + saturation_loss(probe).sum() * self.sat_scale
                pixel_grad = pixel_grad + torch.autograd.grad(
                    term, probe, retain_graph=(k < len(starts) - 1))[0]

            # Stage two: one pass back through the decoder.
            grad = torch.autograd.grad(pixels, x, grad_outputs=pixel_grad)[0]
        return self.clamp(grad)

    def clamp(self, grad):
        """Bound the step size. Without this a single confident step wrecks the image."""
        if not self.clamp_max:
            return grad
        magnitude = grad.square().mean(dim=list(range(1, grad.dim())), keepdim=True).sqrt()
        magnitude = torch.nan_to_num(magnitude, nan=0.0)
        return grad * magnitude.clamp(max=self.clamp_max) / magnitude.clamp(min=1e-12)

"""Command line entry point."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from .clip_bank import ClipBank, DEFAULT_MODELS
from .cutouts import MakeCutouts
from .guidance import PromptGuidance


def build_clip(args, device):
    if args.clip_models:
        names = [tuple(m.split(':')) if ':' in m else (m, 'openai')
                 for m in args.clip_models.split(',')]
    else:
        names = DEFAULT_MODELS
    return ClipBank(names, device=device)


def main():
    ap = argparse.ArgumentParser(description='CLIP-guided sampling, Disco Diffusion style')
    ap.add_argument('prompt', nargs='+', help='one or more prompts; weight with "text::0.5"')
    ap.add_argument('--backend', default='latent-flow', choices=['latent-flow', 'pixel'])
    ap.add_argument('--config', help='latent-flow: LightningDiT config yaml')
    ap.add_argument('--ckpt', help='latent-flow: checkpoint .pt')
    ap.add_argument('--source', default='ema', choices=['ema', 'model'])
    ap.add_argument('--class-id', type=int, default=None, help='optional class condition')
    ap.add_argument('--cfg-scale', type=float, default=1.0)
    ap.add_argument('--out', default='out.png')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--steps', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    # the knobs that shape the look
    ap.add_argument('--strength', type=float, default=1.0,
                    help='guidance strength relative to the model velocity')
    ap.add_argument('--clip-scale', type=float, default=1.0)
    ap.add_argument('--tv-scale', type=float, default=0.0)
    ap.add_argument('--range-scale', type=float, default=0.0)
    ap.add_argument('--sat-scale', type=float, default=0.0)
    ap.add_argument('--clamp-max', type=float, default=0.0)
    ap.add_argument('--overview-cuts', type=int, default=4)
    ap.add_argument('--inner-cuts', type=int, default=16)
    ap.add_argument('--inner-size-pow', type=float, default=0.5)
    ap.add_argument('--cut-batch', type=int, default=0,
                    help='CLIP cutouts per group; lower it if you run out of memory')
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--clip-models', default=None,
                    help='comma list, e.g. "ViT-B-32:openai,RN50:openai"')
    ap.add_argument('--guidance-from', type=float, default=0.0)
    ap.add_argument('--guidance-until', type=float, default=1.0)
    args = ap.parse_args()

    device = 'cuda'
    prompts, weights = [], []
    for p in args.prompt:
        if '::' in p:
            text, w = p.rsplit('::', 1)
            prompts.append(text)
            weights.append(float(w))
        else:
            prompts.append(p)
            weights.append(1.0)

    bank = build_clip(args, device)
    cutouts = MakeCutouts(bank.cut_size, overview=args.overview_cuts, inner=args.inner_cuts,
                          inner_size_pow=args.inner_size_pow, augment=not args.no_augment)
    guidance = PromptGuidance(bank, cutouts, prompts, weights,
                              clip_scale=args.clip_scale, tv_scale=args.tv_scale,
                              range_scale=args.range_scale, sat_scale=args.sat_scale,
                              clamp_max=args.clamp_max)

    if args.backend == 'latent-flow':
        from .backends.latent_flow import LatentFlowBackend
        backend = LatentFlowBackend(args.config, args.ckpt, source=args.source, device=device)
        pixels = backend.sample(
            guidance=guidance, batch_size=args.batch_size, steps=args.steps,
            class_id=args.class_id, cfg_scale=args.cfg_scale,
            cfg_interval_start=backend.config['sample'].get('cfg_interval_start', 0.0),
            timestep_shift=backend.config['sample'].get('timestep_shift', 0.0),
            seed=args.seed, guidance_from=args.guidance_from,
            guidance_until=args.guidance_until, guidance_strength=args.strength,
            cut_batch=args.cut_batch)
        images = backend.to_uint8(pixels)
    else:
        raise SystemExit('pixel backend not wired up yet')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if len(images) == 1:
        Image.fromarray(images[0]).save(args.out)
    else:
        grid = np.concatenate(list(images), axis=1)
        Image.fromarray(grid).save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

"""Command line entry point."""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from .clip_bank import ClipBank, DEFAULT_MODELS
from .cutouts import MakeCutouts
from .guidance import PromptGuidance
from . import disco_config


def parse_prompts(items):
    prompts, weights = [], []
    for p in items:
        if '::' in p:
            text, w = p.rsplit('::', 1)
            prompts.append(text)
            weights.append(float(w))
        else:
            prompts.append(p)
            weights.append(1.0)
    return prompts, weights


def main():
    ap = argparse.ArgumentParser(description='Disco Diffusion, on a current GPU')
    ap.add_argument('prompt', nargs='*', help='prompts; weight with "text::0.5"')
    ap.add_argument('--disco-config', help='a Disco Diffusion settings .json; flags override it')
    ap.add_argument('--image-size', type=int, choices=[256, 512], help='which checkpoint')
    ap.add_argument('--ckpt', help='path to the .pt; defaults to weights/disco/<name>')
    ap.add_argument('--width', type=int)
    ap.add_argument('--height', type=int)
    ap.add_argument('--out', default='out.png')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--skip-steps', type=int)
    ap.add_argument('--eta', type=float, help='0 = DDIM, 1 = DDPM, Disco used 0.8')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--strength', type=float, default=3.0,
                    help='guidance strength relative to the step noise level')
    ap.add_argument('--clip-scale', type=float)
    ap.add_argument('--tv-scale', type=float)
    ap.add_argument('--range-scale', type=float)
    ap.add_argument('--sat-scale', type=float)
    ap.add_argument('--clamp-max', type=float)
    ap.add_argument('--overview-cuts', dest='cut_overview', help='count or Disco schedule string')
    ap.add_argument('--inner-cuts', dest='cut_innercut', help='count or Disco schedule string')
    ap.add_argument('--inner-grey-p', dest='cut_icgray_p', help='fraction or schedule string')
    ap.add_argument('--inner-size-pow', type=float)
    ap.add_argument('--cutn-batches', type=int)
    ap.add_argument('--cut-batch', type=int, default=8,
                    help='CLIP cutouts per group; memory only')
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--clip-models', help='comma list, e.g. "ViT-B-32-quickgelu:openai,RN50-quickgelu:openai"')
    ap.add_argument('--clip-denoised', action='store_true')
    ap.add_argument('--secondary', help='path to secondary_model_imagenet_2.pth (default weights/disco/)')
    ap.add_argument('--no-secondary', action='store_true',
                    help='take the CLIP gradient through the UNet instead of the secondary model')
    ap.add_argument('--no-fast-attention', action='store_true',
                    help='use the original einsum attention instead of fused SDPA')
    ap.add_argument('--autocast', choices=['none', 'bf16'], default='bf16',
                    help='run the UNet and secondary model under bf16 autocast (default) or in full fp32')
    ap.add_argument('--fp16', action='store_true',
                    help='half-precision UNet; faster and smaller, but can overflow on wide frames')
    ap.add_argument('--grad-checkpoint', action='store_true')
    args = ap.parse_args()

    # Start from the Disco file if given, then let explicit flags override.
    settings = dict(prompts=[], weights=[], clip_models=None, image_size=512, width=None,
                    height=None, steps=250, skip_steps=0, eta=0.8, seed=0, clip_scale=5000.0,
                    tv_scale=0.0, range_scale=150.0, sat_scale=0.0, clamp_max=0.05,
                    cutn_batches=1, cut_overview=4, cut_innercut=16, cut_icgray_p=0.2,
                    inner_size_pow=0.5, clip_denoised=False, use_secondary=True)
    if args.disco_config:
        settings.update(disco_config.load(args.disco_config))
    for key in ('image_size', 'width', 'height', 'steps', 'skip_steps', 'eta', 'seed',
                'clip_scale', 'tv_scale', 'range_scale', 'sat_scale', 'clamp_max',
                'cutn_batches', 'cut_overview', 'cut_innercut', 'cut_icgray_p',
                'inner_size_pow'):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.clip_denoised:
        settings['clip_denoised'] = True
    if args.prompt:
        settings['prompts'], settings['weights'] = parse_prompts(args.prompt)
    if args.clip_models:
        settings['clip_models'] = [tuple(m.split(':')) if ':' in m else (m, 'openai')
                                   for m in args.clip_models.split(',')]
    if not settings['prompts']:
        ap.error('give at least one prompt, or a --disco-config with text_prompts')

    device = 'cuda'
    bank = ClipBank(settings['clip_models'] or DEFAULT_MODELS, device=device)
    cutouts = MakeCutouts(bank.cut_size, inner_size_pow=settings['inner_size_pow'],
                          augment=not args.no_augment)
    guidance = PromptGuidance(bank, cutouts, settings['prompts'], settings['weights'],
                              clip_scale=settings['clip_scale'], tv_scale=settings['tv_scale'],
                              range_scale=settings['range_scale'], sat_scale=settings['sat_scale'],
                              clamp_max=settings['clamp_max'])

    from .backends.pixel import PixelBackend
    ckpt = args.ckpt or PixelBackend.default_path(settings['image_size'])
    secondary = args.secondary or PixelBackend.default_secondary_path()
    backend = PixelBackend(ckpt, image_size=settings['image_size'], device=device,
                           fp16=args.fp16, use_checkpoint=args.grad_checkpoint,
                           secondary_path=None if args.no_secondary else secondary,
                           fast_attention=not args.no_fast_attention,
                           autocast_dtype=torch.bfloat16 if args.autocast == 'bf16' else None)
    if backend.secondary is None and not args.no_secondary:
        print('secondary model not found; taking the gradient through the UNet instead '
              f'(expected at {secondary})')
    pixels = backend.sample(
        guidance=guidance, batch_size=args.batch_size, steps=settings['steps'],
        seed=settings['seed'], guidance_strength=args.strength, cut_batch=args.cut_batch,
        eta=settings['eta'], width=settings['width'], height=settings['height'],
        cut_overview=settings['cut_overview'], cut_innercut=settings['cut_innercut'],
        cut_icgray_p=settings['cut_icgray_p'], cutn_batches=settings['cutn_batches'],
        clip_denoised=settings['clip_denoised'], skip_steps=settings['skip_steps'],
        use_secondary=settings['use_secondary'] and not args.no_secondary)
    images = backend.to_uint8(pixels)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if len(images) == 1:
        Image.fromarray(images[0]).save(args.out)
    else:
        Image.fromarray(np.concatenate(list(images), axis=1)).save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

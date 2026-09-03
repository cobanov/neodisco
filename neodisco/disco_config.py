"""Read a Disco Diffusion settings file.

Everyone who used Disco has a folder of these JSON files. They are the record of what was
made, and the point of this project is that they should still run. The keys are mapped
onto the sampler here; anything about animation, video init or 3D is ignored, and the
"secondary model" toggle is ignored too, because on a current card the real model's own
prediction is affordable and more accurate.
"""

import json
import re

# Disco's checkbox names -> open_clip (model, pretrained). The OpenAI weights need the
# quickgelu variants; loading them into a plain-GELU tower is a silent quality loss.
CLIP_NAMES = {
    'ViTB32': ('ViT-B-32-quickgelu', 'openai'),
    'ViTB16': ('ViT-B-16-quickgelu', 'openai'),
    'ViTL14': ('ViT-L-14-quickgelu', 'openai'),
    'ViTL14_336': ('ViT-L-14-336-quickgelu', 'openai'),
    'RN50': ('RN50-quickgelu', 'openai'),
    'RN101': ('RN101-quickgelu', 'openai'),
    'RN50x4': ('RN50x4', 'openai'),
    'RN50x16': ('RN50x16', 'openai'),
    'RN50x64': ('RN50x64', 'openai'),
}

_WEIGHT = re.compile(r'^(.*?):\s*(-?\d+(?:\.\d+)?)\s*$')


def split_prompt(text):
    """Disco wrote weights as a trailing ':5'. Returns (text, weight)."""
    m = _WEIGHT.match(text)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return text.strip(), 1.0


def load(path, frame=0):
    """Return a dict of keyword arguments for the pixel backend and guidance."""
    with open(path) as handle:
        cfg = json.load(handle)

    # text_prompts is keyed by the frame the prompt set starts at; take the last set
    # whose start is <= the requested frame, which for a still image is set "0".
    prompt_sets = cfg.get('text_prompts', {})
    starts = sorted(int(k) for k in prompt_sets)
    active = [k for k in starts if k <= frame] or starts[:1]
    raw = prompt_sets[str(active[-1])] if active else []
    prompts, weights = zip(*(split_prompt(p) for p in raw)) if raw else ([], [])

    clip_models = [CLIP_NAMES[k] for k in CLIP_NAMES if cfg.get(k)]
    image_size = 512 if '512' in str(cfg.get('diffusion_model', '512')) else 256

    return dict(
        prompts=list(prompts),
        weights=list(weights),
        clip_models=clip_models or None,
        image_size=image_size,
        width=int(cfg.get('width', image_size)),
        height=int(cfg.get('height', image_size)),
        steps=int(cfg.get('steps', 250)),
        skip_steps=int(cfg.get('skip_steps', 0)),
        eta=float(cfg.get('eta', 0.8)),
        seed=int(cfg['seed']) if cfg.get('seed') not in (None, 'random_seed') else 0,
        clip_scale=float(cfg.get('clip_guidance_scale', 5000)),
        tv_scale=float(cfg.get('tv_scale', 0)),
        range_scale=float(cfg.get('range_scale', 150)),
        sat_scale=float(cfg.get('sat_scale', 0)),
        clamp_max=float(cfg.get('clamp_max', 0.05)) if cfg.get('clamp_grad', True) else 0.0,
        cutn_batches=int(cfg.get('cutn_batches', 1)),
        cut_overview=cfg.get('cut_overview'),
        cut_innercut=cfg.get('cut_innercut'),
        cut_icgray_p=cfg.get('cut_icgray_p'),
        inner_size_pow=float(cfg.get('cut_ic_pow', 1.0)),
        clip_denoised=bool(cfg.get('clip_denoised', False)),
        use_secondary=bool(cfg.get('use_secondary_model', True)),
    )

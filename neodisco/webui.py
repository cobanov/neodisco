"""A small web page for running the original Disco Diffusion models.

    python -m neodisco.webui

Everything the command line exposes is here, with Disco's defaults filled in, plus a
place to drop an old settings file. Runs are queued one at a time, since a single GPU is
the usual case.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from .clip_bank import ClipBank
from .cutouts import MakeCutouts
from .guidance import PromptGuidance
from .backends.pixel import PixelBackend
from . import disco_config

CLIP_CHOICES = list(disco_config.CLIP_NAMES.keys())
_cache = {}


def _backend(image_size, weights_dir, fp16):
    key = (image_size, fp16)
    if key not in _cache:
        for k in list(_cache):
            del _cache[k]
        torch.cuda.empty_cache()
        _cache[key] = PixelBackend(
            PixelBackend.default_path(image_size, weights_dir), image_size=image_size,
            fp16=fp16, use_checkpoint=True,
            secondary_path=PixelBackend.default_secondary_path(weights_dir))
    return _cache[key]


def _bank(names):
    key = ('clip', tuple(names))
    if key not in _cache:
        _cache[key] = ClipBank([disco_config.CLIP_NAMES[n] for n in names])
    return _cache[key]


def generate(prompt_text, settings_file, image_size, width, height, steps, skip_steps, seed,
             eta, clamp_max, clip_scale, tv_scale, range_scale, sat_scale, cutn_batches,
             cut_overview, cut_innercut, cut_icgray_p, inner_size_pow, clip_names,
             use_secondary, fp16, weights_dir, progress=None):
    import gradio as gr
    settings = dict(prompts=[], weights=[])
    if settings_file:
        settings.update(disco_config.load(settings_file))
    if prompt_text.strip():
        prompts, weights = [], []
        for line in prompt_text.splitlines():
            line = line.strip()
            if not line:
                continue
            text, w = disco_config.split_prompt(line)
            prompts.append(text)
            weights.append(w)
        settings['prompts'], settings['weights'] = prompts, weights
    if not settings['prompts']:
        raise gr.Error('Write a prompt, or drop in a settings file that has one.')

    names = list(clip_names) or ['ViTB32', 'ViTB16', 'RN50']
    bank = _bank(names)
    cutouts = MakeCutouts(bank.cut_size, inner_size_pow=float(inner_size_pow))
    guidance = PromptGuidance(bank, cutouts, settings['prompts'], settings['weights'],
                              clip_scale=float(clip_scale), tv_scale=float(tv_scale),
                              range_scale=float(range_scale), sat_scale=float(sat_scale),
                              clamp_max=float(clamp_max))
    backend = _backend(int(image_size), weights_dir, bool(fp16))
    seed = int(seed) if int(seed) >= 0 else int(torch.randint(0, 2 ** 31, ()))

    def sched(v):
        v = str(v).strip()
        return v if v else None

    if progress is not None:
        progress(0, desc='sampling')
    pixels = backend.sample(
        guidance=guidance, steps=int(steps), seed=seed, width=int(width), height=int(height),
        eta=float(eta), skip_steps=int(skip_steps), cut_overview=sched(cut_overview),
        cut_innercut=sched(cut_innercut), cut_icgray_p=sched(cut_icgray_p),
        cutn_batches=int(cutn_batches), cut_batch=16, use_secondary=bool(use_secondary),
        progress=False)
    image = Image.fromarray(backend.to_uint8(pixels)[0])
    used = dict(settings, seed=seed, width=int(width), height=int(height), steps=int(steps),
                eta=float(eta), clamp_max=float(clamp_max), clip_models=names)
    used.pop('weights', None)
    return image, json.dumps(used, indent=2, ensure_ascii=False)


def build(weights_dir):
    import gradio as gr
    with gr.Blocks(title='neodisco') as demo:
        gr.Markdown('## neodisco\nDisco Diffusion, the original models, on a current GPU. '
                    'One prompt per line; weight a line with `:2` at the end. '
                    'Or drop in an old Disco settings file and press Generate.')
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label='Prompts', lines=4,
                                    value='a fractal cathedral of glowing coral, intricate, dreamlike, trending on artstation')
                settings_file = gr.File(label='Disco settings .json (optional)', file_types=['.json'], type='filepath')
                with gr.Row():
                    image_size = gr.Radio([256, 512], value=512, label='Checkpoint')
                    width = gr.Number(value=1280, precision=0, label='Width (multiple of 64)')
                    height = gr.Number(value=768, precision=0, label='Height')
                with gr.Row():
                    steps = gr.Slider(50, 500, value=250, step=10, label='Steps')
                    skip_steps = gr.Slider(0, 100, value=10, step=1, label='Skip steps')
                    seed = gr.Number(value=-1, precision=0, label='Seed (-1 = random)')
                with gr.Accordion('Guidance', open=True):
                    with gr.Row():
                        clamp_max = gr.Slider(0.0, 0.2, value=0.05, step=0.005, label='clamp_max (strength)')
                        eta = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label='eta')
                        cutn_batches = gr.Slider(1, 8, value=4, step=1, label='cutn_batches')
                    with gr.Row():
                        clip_scale = gr.Number(value=5000, label='clip_guidance_scale')
                        tv_scale = gr.Number(value=0, label='tv_scale')
                        range_scale = gr.Number(value=150, label='range_scale')
                        sat_scale = gr.Number(value=0, label='sat_scale')
                    clip_names = gr.CheckboxGroup(CLIP_CHOICES, value=['ViTB32', 'ViTB16', 'RN50'], label='CLIP models')
                with gr.Accordion('Cutouts', open=False):
                    cut_overview = gr.Textbox(value='[12]*400+[4]*600', label='cut_overview')
                    cut_innercut = gr.Textbox(value='[4]*400+[12]*600', label='cut_innercut')
                    cut_icgray_p = gr.Textbox(value='[0.2]*400+[0]*600', label='cut_icgray_p')
                    inner_size_pow = gr.Number(value=1.0, label='cut_ic_pow')
                with gr.Accordion('Advanced', open=False):
                    use_secondary = gr.Checkbox(value=True, label='Use secondary model (Disco default)')
                    fp16 = gr.Checkbox(value=False, label='fp16 UNet (can overflow on wide frames)')
                run = gr.Button('Generate', variant='primary')
            with gr.Column(scale=1):
                out = gr.Image(label='Result', type='pil')
                used = gr.Code(label='Settings used (paste back into a .json)', language='json')
        weights_state = gr.State(weights_dir)
        run.click(generate,
                  inputs=[prompt, settings_file, image_size, width, height, steps, skip_steps, seed,
                          eta, clamp_max, clip_scale, tv_scale, range_scale, sat_scale, cutn_batches,
                          cut_overview, cut_innercut, cut_icgray_p, inner_size_pow, clip_names,
                          use_secondary, fp16, weights_state],
                  outputs=[out, used])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/disco')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true')
    args = ap.parse_args()
    demo = build(args.weights)
    demo.queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=args.share,
                                  show_error=True)


if __name__ == '__main__':
    main()

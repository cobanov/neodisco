"""Independent source oracles for the supported still-image Disco path."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from neodisco._resize_right import resize
from neodisco.backends.pixel import build_diffusion
from neodisco.backends.secondary import alpha_sigma_to_t
from neodisco.cutouts import MakeCutouts
from neodisco.schedules import parse_schedule
from neodisco.settings import normalise_settings, effective_record
from test_guidance import guidance
from test_sampler import toy_backend, TinySecondary

REFERENCE = Path(__file__).parent / 'reference'


def original(**overrides):
    ns = dict(torch=torch, nn=nn, F=F, T=T, TF=TF, resize=resize,
              args=SimpleNamespace(animation_mode='None'), padargs={},
              skip_augs=False, cutout_debug=False, device=torch.device('cpu'),
              alpha_sigma_to_t=alpha_sigma_to_t)
    ns.update(overrides)
    exec(compile((REFERENCE / 'disco.py').read_text(), 'original_disco.py', 'exec'), ns)
    return ns


@pytest.mark.parametrize('overview', [4, 12])
@pytest.mark.parametrize('augment', [False, True])
def test_cutouts_pixels_and_gradient_match_saved_upstream_oracle(overview, augment):
    # Golden tensors were generated with the independent original ResizeRight checkout,
    # original Dango class, fixed CPU input and RNG. No modern implementation in oracle.
    saved = np.load(REFERENCE / 'cutouts.npz')
    x = torch.linspace(-1.3, 1.2, 3 * 40 * 64).reshape(1, 3, 40, 64).requires_grad_()
    cuts = MakeCutouts(16, overview=overview, inner=3, inner_size_pow=1,
                       inner_grey_p=.2, augment=augment)
    torch.manual_seed(123)
    y = cuts(x)
    grad = torch.autograd.grad(y.square().sum(), x)[0]
    key = f'{overview}_{int(augment)}'
    torch.testing.assert_close(y, torch.from_numpy(saved[key + '_cuts']), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(grad, torch.from_numpy(saved[key + '_grad']), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('secondary', [False, True])
@pytest.mark.parametrize('chunk', [1, 4, 0])
def test_score_matches_source_cond_fn_and_rng_stream(secondary, chunk):
    guide = guidance()
    guide.bank.sizes = [4, 6]  # models must get independently drawn native-size cuts
    for i, model in enumerate(guide.bank.models):
        model.visual = SimpleNamespace(input_resolution=guide.bank.sizes[i])
    guide.cutouts = MakeCutouts(6, overview=4, inner=3, inner_size_pow=1)
    guide.range_scale, guide.tv_scale, guide.sat_scale = 150, .3, .2
    guide.clamp_max = .05
    diffusion = build_diffusion('ddim4')
    backend = toy_backend()
    secondary_model = TinySecondary()
    args = SimpleNamespace(animation_mode='None', cutn_batches=2,
                           cut_overview=[4]*1000, cut_innercut=[3]*1000,
                           cut_ic_pow=[1]*1000, cut_icgray_p=[.2]*1000,
                           clamp_grad=True, clamp_max=.05)
    # A secondary prediction outside [-1,1] makes the zero range-derivative distinction observable.
    x = torch.linspace(-3, 3, 3*12*20).reshape(1, 3, 12, 20)
    ref = original(args=args, use_secondary_model=secondary, diffusion=diffusion,
                   cur_t=1, secondary_model=secondary_model,
                   model=lambda x, t, **kw: backend.model(x, t),
                   model_stats=[{'clip_model': m, 'target_embeds': e, 'weights': guide.weights}
                                for m, e in zip(guide.bank.models, guide.embeddings)],
                   normalize=lambda x: x, loss_values=[], clip_guidance_scale=guide.clip_scale,
                   cutn_batches=2, tv_scale=.3, range_scale=150, sat_scale=.2,
                   init=None, init_scale=0)
    torch.manual_seed(55)
    expected = ref['cond_fn'](x, torch.tensor([250.]))
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(55)
    probe = x.detach().requires_grad_()
    sigma = float(diffusion.sqrt_one_minus_alphas_cumprod[1])
    if secondary:
        pred = secondary_model(probe, None).pred
    else:
        pred = diffusion.p_mean_variance(backend.model, probe, torch.tensor([1]),
                                        clip_denoised=False)['pred_xstart']
    blend = pred * sigma + probe * (1 - sigma)
    image_grad = guide.image_gradient(blend, cut_batch=chunk, cutn_batches=2,
                                     range_target=lambda _: pred.detach())
    actual = -guide.clamp(torch.autograd.grad(blend, probe, image_grad)[0])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-5)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.parametrize('steps', [80, 120, 160, 240, 250, 333])
def test_sampler_uses_notebook_base_schedule_and_scaled_model_time(steps):
    backend = toy_backend()
    seen = []
    class Capture(nn.Module):
        def forward(self, x, t):
            seen.append(float(t[0]))
            return torch.cat([x*.05, torch.zeros_like(x)], 1)
    backend.model = backend._compiled_model = Capture()
    backend.sample(steps=steps, skip_steps=steps-2, seed=7, width=64, height=64,
                   progress=False)
    # Verify a nonzero time actually reaches the model after rescaling.
    n = (1000//steps)*steps
    assert seen == [float(np.float32(n//steps)*np.float32(1000/n)), 0.]
    d = build_diffusion(f'ddim{steps}', diffusion_steps=n)
    assert d.timestep_map == list(range(0, n, n//steps))
    assert d.rescale_timesteps
    np.testing.assert_allclose(d.betas[0], .0001 * 1000/n, rtol=1e-10)


def test_skip_without_init_and_saved_prediction_match_reference():
    backend = toy_backend()
    previews = []
    result = backend.sample(steps=4, skip_steps=2, seed=77, eta=.8,
                            width=64, height=64, progress=False,
                            preview=lambda n, x: previews.append(x.clone()))
    d = build_diffusion('ddim4')
    torch.manual_seed(77)
    noise = torch.randn(1, 3, 64, 64)
    x = d.q_sample(torch.zeros_like(noise), torch.tensor([1]), noise=noise)
    for i in [1, 0]:
        out = d.ddim_sample(backend.model, x, torch.tensor([i]), clip_denoised=False, eta=.8)
        x = out['sample']
    torch.testing.assert_close(result, out['pred_xstart_uncond'], atol=0, rtol=0)
    torch.testing.assert_close(result, previews[-1], atol=0, rtol=0)


def test_guided_ddim_exposes_original_unconditioned_save_without_changing_step():
    d = build_diffusion('ddim4')
    model = toy_backend().model
    x = torch.linspace(-.8, .9, 48).reshape(1, 3, 4, 4)
    t = torch.tensor([2])
    p = d.p_mean_variance(model, x, t, clip_denoised=False)
    score = torch.full_like(x, .05)
    a = float(d.alphas_cumprod[2]); prev = float(d.alphas_cumprod_prev[2])
    # Original condition_score and DDIM equations, independently expanded.
    eps = (x - a**.5 * p['pred_xstart']) / (1-a)**.5
    eps = eps - (1-a)**.5 * score
    pred = (x - (1-a)**.5 * eps) / a**.5
    expected = prev**.5 * pred + (1-prev)**.5 * eps
    out = d.ddim_sample(model, x, t, clip_denoised=False, cond_fn=lambda x, t: score, eta=0, model_kwargs={})
    torch.testing.assert_close(out['sample'], expected)
    torch.testing.assert_close(out['pred_xstart_uncond'], p['pred_xstart'])
    assert not torch.equal(out['pred_xstart'], out['pred_xstart_uncond'])


def test_original_power_schedule_and_effective_semantics_are_preserved():
    s = normalise_settings({'text_prompts': {'0': ['blue']},
                            'cut_ic_pow': '[1]*400+[2]*600'})
    assert s['inner_size_pow'] == '[1]*400+[2]*600'
    assert parse_schedule(s['inner_size_pow'], 1000)[400] == 2
    assert effective_record(s)['sampling_semantics'] == 'disco-2026-09-07'


@pytest.mark.parametrize('secondary', [False, True])
def test_complete_pixel_backend_matches_original_guidance_loop(secondary):
    backend = toy_backend()
    backend.secondary = TinySecondary() if secondary else None
    guide = guidance()
    guide.cutouts = MakeCutouts(4, inner_size_pow=1)
    guide.range_scale = 150
    guide.clamp_max = .05
    for model in guide.bank.models:
        model.visual = SimpleNamespace(input_resolution=4)
    overview = [4]*400 + [1]*600
    inner = [1]*400 + [3]*600
    power = [1]*400 + [2]*600
    args = SimpleNamespace(animation_mode='None', cutn_batches=2,
                           cut_overview=overview, cut_innercut=inner,
                           cut_ic_pow=power, cut_icgray_p=[.2]*1000,
                           clamp_grad=True, clamp_max=.05)
    d = build_diffusion('ddim4')
    ref = original(args=args, use_secondary_model=secondary, diffusion=d,
                   cur_t=2, secondary_model=backend.secondary,
                   model=lambda x, t, **kw: backend.model(x, t),
                   model_stats=[{'clip_model': m, 'target_embeds': e, 'weights': guide.weights}
                                for m, e in zip(guide.bank.models, guide.embeddings)],
                   normalize=lambda x: x, loss_values=[], clip_guidance_scale=guide.clip_scale,
                   cutn_batches=2, tv_scale=0, range_scale=150, sat_scale=0,
                   init=None, init_scale=0)
    torch.manual_seed(77)
    noise = torch.randn(1, 3, 64, 64)
    x = d.q_sample(torch.zeros_like(noise), torch.tensor([2]), noise=noise)
    for i in [2, 1, 0]:
        ref['cur_t'] = i
        out = d.ddim_sample(backend.model, x, torch.tensor([i]), clip_denoised=False,
                            cond_fn=ref['cond_fn'], eta=.8, model_kwargs={})
        x = out['sample']
    expected = out['pred_xstart_uncond']
    actual = backend.sample(guidance=guide, steps=4, skip_steps=1, seed=77,
                            eta=.8, width=64, height=64, progress=False, cut_batch=8,
                            cut_overview=overview, cut_innercut=inner, cut_ic_pow=power,
                            cut_icgray_p=.2, cutn_batches=2, use_secondary=secondary)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_cut_schedules_index_the_original_rescaled_time():
    from test_sampler import InitRecorder
    class CaptureGuide(InitRecorder):
        def image_gradient(self, pixels, **kw):
            self.calls.append((kw['overview'], kw['inner_size_pow']))
            return torch.zeros_like(pixels)
    guide = CaptureGuide()
    values = list(range(1000))
    toy_backend().sample(guidance=guide, steps=120, skip_steps=117, seed=3,
                         width=64, height=64, progress=False,
                         cut_overview=values, cut_ic_pow=[v+1 for v in values])
    indices = [999-int(np.float32(i*8)*np.float32(1000/960)) for i in [2,1,0]]
    assert guide.calls[1:] == [(i,i+1) for i in indices]


def test_original_skip_augs_is_imported():
    s = normalise_settings({'text_prompts': {'0': ['blue']}, 'skip_augs': True})
    assert s['augment'] is False


@pytest.mark.parametrize('extra', [
    {'diffusion_sampling_mode': 'plms'}, {'perlin_init': True},
    {'fuzzy_prompt': True}, {'image_prompts': {'0': ['image.png']}},
])
def test_unsupported_original_generation_modes_are_not_silently_reinterpreted(extra):
    with pytest.raises(ValueError):
        normalise_settings({'text_prompts': {'0': ['blue']}, **extra})


def test_resize_plan_cache_matches_original_and_does_not_keep_input_graphs():
    from neodisco.resize import ResizeRightPlans
    cache = ResizeRightPlans(max_entries=2, max_bytes=2048)
    for height, width, size in [(12,20,4), (13,19,6), (8,8,16), (12,20,4)]:
        x = torch.linspace(-2,2,3*height*width).reshape(1,3,height,width).requires_grad_()
        y = cache(x,size)
        expected = resize(x, out_shape=[1,3,size,size])
        torch.testing.assert_close(y, expected, atol=0, rtol=0)
        a = torch.autograd.grad(y.square().sum(),x)[0]
        b = torch.autograd.grad(expected.square().sum(),x)[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        assert len(cache.plans)<=2 and cache.bytes<=2048
        assert all(not t.requires_grad and t.grad_fn is None
                   for p in cache.plans.values() for t in p[:2])
    with torch.inference_mode():
        cache(torch.zeros(1,3,17,17),4)
    x = torch.ones(1,3,17,17,requires_grad=True)
    cache(x,4).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_original_nan_guidance_recovery_is_counted_and_does_not_mask_model_failure():
    from test_sampler import InitRecorder, NaNUNet
    class NaNGuide(InitRecorder):
        def image_gradient(self, pixels, **kw):
            return torch.full_like(pixels, float('nan'))
    b = toy_backend()
    with pytest.warns(RuntimeWarning, match='skipping affected steps'):
        actual = b.sample(guidance=NaNGuide(), steps=4, skip_steps=1, seed=4,
                          width=64, height=64, progress=False)
    expected = toy_backend().sample(steps=4, skip_steps=1, seed=4,
                                    width=64, height=64, progress=False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert b.guidance_nan_steps == [1,2,3]
    b.sample(steps=4, seed=4, width=64, height=64, progress=False)
    assert b.guidance_nan_steps == []
    with pytest.raises(FloatingPointError, match='non-finite'):
        toy_backend(NaNUNet()).sample(guidance=NaNGuide(), steps=4, seed=4,
                                      width=64, height=64, progress=False)


def test_source_cond_fn_returns_zero_for_nan_image_gradient():
    g = guidance()
    for model in g.bank.models:
        model.visual = SimpleNamespace(input_resolution=4)
        model.weight.data.fill_(float('nan'))
    args = SimpleNamespace(animation_mode='None', cutn_batches=1,
                           cut_overview=[1]*1000, cut_innercut=[0]*1000,
                           cut_ic_pow=[1]*1000, cut_icgray_p=[0]*1000,
                           clamp_grad=True, clamp_max=.05)
    ns = original(args=args, use_secondary_model=True, diffusion=build_diffusion('ddim4'),
                  cur_t=1, secondary_model=TinySecondary(),
                  model_stats=[{'clip_model':m,'target_embeds':e,'weights':g.weights}
                               for m,e in zip(g.bank.models,g.embeddings)],
                  normalize=lambda x:x, loss_values=[], clip_guidance_scale=2,
                  cutn_batches=1,tv_scale=0,range_scale=150,sat_scale=0,init=None,init_scale=0)
    x = torch.ones(1,3,8,8)
    assert torch.equal(ns['cond_fn'](x,torch.tensor([250.])),torch.zeros_like(x))

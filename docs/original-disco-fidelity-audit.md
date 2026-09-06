# Original Disco Diffusion fidelity audit

Date: 2026-09-07. Audited neodisco release: `95acbf46b599cdedf5d852ead8e012e965deabb4`.

**The current port is not numerically or stochastically equivalent to original Disco.**
The modernization tests established stability, internal consistency and repeatability
on the tested runtime. They did not establish original-notebook equivalence. Several
differences below predate that modernization. This audit does not attribute every visual
difference to a single bug, and does not change application code or the live deployment.

The user prefers 240 actual sampling iterations. The example used here is `steps=250`,
`skip_steps=10`, rather than `steps=240`, `skip_steps=0`; those use different schedules.

## Sources and reproduction scope

- [Disco Diffusion](https://github.com/alembics/disco-diffusion/blob/37eb39bfe0e7310c86c244859b789a5346754251/disco.py), commit `37eb39bfe0e7310c86c244859b789a5346754251`.
- [The sampler fork actually cloned by Disco](https://github.com/kostarion/guided-diffusion/blob/ced66cb5b4dc58580701cc1519f5efd3b0436b27/guided_diffusion/gaussian_diffusion.py), commit `ced66cb5b4dc58580701cc1519f5efd3b0436b27`.
- [ResizeRight](https://github.com/assafshocher/ResizeRight/blob/510d4d5b67dccf4efdee9f311ed42609a71f17c5/resize_right.py), commit `510d4d5b67dccf4efdee9f311ed42609a71f17c5`.
- Original OpenAI CLIP [loader](https://github.com/openai/CLIP/blob/main/clip/clip.py) and [model/weight conversion](https://github.com/openai/CLIP/blob/main/clip/model.py).

The supplied `TimeToDisco(0)_14.png` is 1280x768 RGB and contains no PNG metadata.
The exact historical notebook version and complete run settings are not yet known.
The repository example is a reduced config. The main notebook and archived v4.1 code
were both checked for the range-gradient and per-model cutout behavior.

Local raw evidence is under `benchmark-results/disco-fidelity-audit/` (gitignored):
upstream checkouts, extracted notebook code, `numerical-audit.json`, `ablation.py`,
source-extracted `original_cutouts.py`, GPU outputs and manifest.

## Confirmed differences

### 1. Range regularization follows a different derivative path

Original `disco.py:1352-1405` constructs `x_in = pred * sigma + x * (1-sigma)`,
evaluates the range penalty on `pred`, then asks autograd for its derivative with
respect to `x_in`. Since `pred` is upstream of `x_in`, that range term contributes
zero to this derivative. The TV and saturation terms keep the requested graph valid.
This is a quirk, likely a bug, of the reference implementation.

Our `backends/pixel.py:310-312` reconstructs the prediction from a detached `x` and
the gradient probe, then `guidance.py:144-149` differentiates through that reconstruction.
This creates a nonzero range gradient absent in the original. It also is not simply
the true derivative of `range(pred)` with respect to `x`: the reconstruction introduces
an extra derivative path when the blended-image gradient is propagated back.

CPU autograd check, `x=[3,-3,2,-2]`, `pred=0.8*x`, `sigma=0.5`, `range_scale=150`:
original derivative with respect to `x_in` is `[0,0,0,0]`; port derivative is
approximately `[210,-210,90,-90]`. This is an algebraic compatibility issue, not rounding.

For original behavior, range regularization must have the reference derivative semantics.
A deliberately repaired regularizer should be a separately documented behavior.

### 2. CLIP models share random cutouts that were independent in Disco

Original `disco.py:1371-1392` loops over models, then over `cutn_batches`, drawing
fresh cutouts each time. With three CLIP models and four draws, this means twelve
independent cutout sets per iteration.

Our `guidance.py:123-138` draws four sets and reuses them across the three models.
The averaging and model sum are correct for a single image, but reusing random
samples changes gradient correlations and RNG consumption. Equality of expected losses
does not imply equality of sampled gradients or generated images. Batching already
drawn independent cutouts remains a potential optimization.

### 3. Resizing is not the same operation

Original `MakeCutoutsDango` uses ResizeRight (`disco.py:1017,1030,1049`). Our
`cutouts.py:48-50` uses PyTorch bicubic antialias interpolation. These are different
kernels/boundary implementations. The separate Lanczos helper in the notebook is
not the resize used by this Dango path.

With the supplied image, matching crop RNG seed 123, 12 overview plus 4 inner cuts,
and augmentations disabled: cutout RMSE is `0.0033381293`, maximum absolute difference
`0.09594250`, and the relative L2 difference of the squared-cutout-loss image gradient
is `0.009566669` (0.957%). This demonstrates a numerical difference, not a measured
perceptual quality loss or a proof that this alone causes dark images.

### 4. Precision policies differ

Original CUDA OpenAI CLIP loads mixed fp16 weights; `.float()` on returned embeddings
does not make the encoder fp32. Original secondary inference is fp32, and the UNet
uses its fp16 conversion path. Our tested configuration uses fp32 CLIP and bf16
UNet/secondary forwards. OpenCLIP and SDPA also introduce implementation differences.

The current fp32 CLIP policy is a stability choice, not a faithful recreation of
historical CUDA precision. The GPU experiment below holds these modern choices fixed;
it cannot quantify the separate effect of precision or establish full original parity.

### 5. Seed identifies a stream, not every image in a notebook batch

Original `disco.py:1263-1268` seeds before the `n_batches` loop, without reseeding
between still-image batches. The filename suffix `_14` is consistent with the fifteenth
image in a still-image batch (or an animation frame in another mode). It is not proof
that it was the first image generated from seed 1290357248.

Our render resets the RNG for each image. Even after fixing all arithmetic, reproducing
the fifteenth historical result requires the appropriate RNG stream position as well
as the complete settings, model weights, sampler and runtime. The filename alone does
not establish those details.

### 6. Smaller sampler differences and schedule limitations

- Disco's kostarion sampler returns the unguided `pred_xstart` for display/save;
  neodisco uses the guided prediction for previews and returns the final sample.
  At final timestep zero, with this schedule and gradient RMS bounded by 0.05,
  the clean-prediction difference is at most about `5e-6` RMS in model space.
  This does not explain a large final tonal shift. Earlier previews can differ more.
- With `skip_steps>0` and no init image, the original fork initializes from a
  zero image noised to the start timestep. Our port uses raw noise. For this
  250/10 configuration the reference noise multiplier is `0.99995277`, so the
  direct difference is small; the stochastic trajectory can still diverge.
- Original arbitrary step counts use `(1000//steps)*steps` base diffusion steps
  followed by `ddim{steps}`. Our port keeps 1000 base steps and falls back to even
  spacing for non-divisors. The previous 80/120/160/240 sweep measures this port,
  not the notebook's schedules. The 250/10 configuration is unaffected by this issue.
- The original cutout schedule indexes `999-original_t`. Our resampled schedule
  is not generally identical. For the example's 250-step schedule and 400/600
  boundaries, overview/inner/grey values do match at each sampled step. General
  `cut_ic_pow` schedules are also not represented by our scalar-only setting.

## What matches in the audited single-image path

The 512 UNet architecture configuration, secondary-model prediction/blend structure,
spherical CLIP distance, prompt normalization for these unweighted prompts, sum over
CLIP models, per-draw averaging, gradient sign, RMS clamp and main DDIM update formula
match at the algebraic level. Floating-point implementation differences remain.
All ten function/class definitions in `backends/secondary.py`, including both secondary
networks and their helpers, also match the original AST exactly (ignoring source locations).

`clip_guidance_scale=5000` is external CLIP loss guidance, not classifier-free guidance
of the kind commonly described as CFG 7.5. No extra global multiplier was found in the
current clamp/update path. Lowering this value is not a substitute for compatibility fixes.

## GPU ablation

All three full renders completed successfully. All variants use the same
example config, seed 1290357248, 1280x768, 250 configured steps minus 10 skipped,
CLIP scale 5000, eta 0.8, secondary model, RTX 5090, fp32 CLIP, bf16 backbones,
SDPA and warm compile. A two-step warmup precedes the timed renders.

The reference-guidance variant restores original range derivative behavior and runs
the source-extracted Dango class with ResizeRight and independent per-model draws.
It is not a full original notebook execution. Each condition has one render; ordinary
CUDA backward nondeterminism remains enabled. Brightness statistics are descriptive,
not a perceptual equivalence score or a statistical attribution of causality.

| Image / condition | Render seconds | Mean luma | Pixels with luma < 0.1 |
|---|---:|---:|---:|
| Supplied historical image | unknown | 0.4197 | 4.36% |
| Current port | 99.61 | 0.3694 | 9.38% |
| Original range derivative only | 99.79 | 0.3809 | 8.73% |
| Original range derivative + original Dango/ResizeRight + independent CLIP draws | 238.49 | 0.3909 | 10.92% |

Statistics above use the saved PNGs consistently, with weighted encoded RGB
`0.2126 R + 0.7152 G + 0.0722 B`, not linear-light physical luminance. Raw tensor
statistics in the manifest's run rows precede uint8 quantization and differ slightly.

Removing the extra range gradient increased mean luma by 0.0115 in this render, but
did not restore the historical soft/grainy appearance. Restoring the cutout behavior
produced a different composition and brighter cloud areas, but retained sharp detail
and substantial dark regions. Its dark-pixel fraction actually increased despite
higher mean luma. The historical image remains visually different in all three cases.
There is no evidence here for a single brightness correction or a universal CFG fix.

The 238.49-second source-loop variant is a deliberately direct oracle candidate, not
an optimized replacement or a prediction of eventual compatible performance. The
extra ResizeRight work and small sequential CLIP batches are bundled in this timing;
their individual costs were not isolated. Compile was held constant across conditions,
so this experiment does not establish compile's independent effect on image appearance.

Local images: `benchmark-results/disco-fidelity-audit/{current,range-original,reference-guidance,comparison}.png`.
`manifest.json` includes hashes, environment and effective experiment semantics.
The sidecars annotate the runtime patch; replaying reference-guidance through the
ordinary CLI alone does not reproduce those semantics. Use `ablation.py`.

## Recommended implementation order

1. Establish an explicitly pinned original-compatibility baseline: reference derivative
   graph, independent per-model random draws, ResizeRight, schedules and sampler outputs.
2. Add small oracle comparisons against source-extracted original functions for
   image/score gradients, cutouts, timestep maps and one DDIM step. Keep fixed RNG inputs
   separate from stochastic full-render comparisons. Internal self-consistency tests
   alone cannot catch the discrepancies found here.
3. Validate a fixed-seed image suite against that baseline, then reintroduce batching,
   modern precision, SDPA and compile independently. Report visual and speed tradeoffs.
4. For exact historical-image investigation, obtain the complete saved settings,
   notebook version, sampler and batch/frame position. Do not promise pixel identity
   across historical GPU/framework environments.

No application fix or deployment was performed as part of this audit.

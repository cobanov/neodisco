# neodisco modernization implementation report

Date: 2026-09-06

2026-09-07 audit addendum: the tests below establish modern-runtime stability and
internal consistency, not equivalence to the original Disco notebook. The
[original Disco fidelity audit](original-disco-fidelity-audit.md) found unresolved
range-gradient, random-cutout, resize and schedule differences. Read that audit
before treating the numerical-correctness phase below as original-compatibility acceptance.
The subsequent [compatibility implementation](disco-compatibility-implementation.md)
adds source oracles and fixes the supported still-image DDIM path.

Baseline revision: `03d5792d75c7d68adab90b414047b4bedb6e99e7`

This report separates CPU verification, real CUDA evidence and measured limitations.
A historical timing is never used as evidence for the corrected sampler.

## Status by phase

| Phase | Status | Evidence |
|---|---|---|
| Test and measurement harness | Complete locally | 48 CPU tests pass; benchmark CLI emits partial and final manifests |
| Numerical correctness and RNG | Complete on tested CUDA stack | all eight 256/512 precision and secondary combinations are finite and repeat exactly in strict mode |
| Settings and resource management | Complete locally | config/API/cache/attention tests pass |
| Corrected performance work | Complete with opt-in compile | compiled steady render time was 24.38% shorter; matched quality checks found no gross compiler-only failure but do not establish perceptual equivalence |
| Package and documentation | Complete locally | frozen lock, CPU CI, wheel build and fresh wheel import verified |

## Correctness changes

- CLIP text encoding, image encoding and guidance loss accumulation explicitly disable
  outer autocast and use fp32. AMP now surrounds only UNet and secondary forwards.
- `auto` precision selects bf16 only on a supporting selected CUDA device. Forced bf16
  and fp16 on CPU fail before model loading. fp16 remains opt-in.
- One serialized render owns CPU and selected-device RNG state and restores caller state.
  Initial noise, DDIM noise, crop geometry and augmentations therefore use one seed.
- CUDA interpolation backward remains nondeterministic in ordinary mode. Strict reference
  mode retains the same transformations but performs differentiable cutout augmentation
  and mixed-size CLIP resizing on CPU. It also enables deterministic PyTorch algorithms.
- Original Disco prompt weights are divided by the absolute signed sum. A sum below
  `0.001` is rejected. Direct library users of `PromptGuidance` retain supplied weights.
- LPIPS contributes `lpips(x_in, init).sum() * init_scale` once per guidance pass in the
  original `[-1, 1]` blended image domain. It is independent of cut chunks and draws.
  A requested nonzero scale without LPIPS is an actionable error.
- `PromptGuidance.loss` and `image_gradient` now both sum CLIP models. The earlier
  average/sum split was inconsistent with Disco and with each other.
- The default `clamp_max > 0` path and the raw unclamped Disco score remain supported.
  The old `guidance_strength` formula belonged to an incorrect hand-written DDIM update,
  then became a silent no-op after switching to reference DDIM. `--strength` is now
  rejected with a migration message instead of inventing new sampler algebra.
- `through_model=False` is rejected by the pixel backend. Supported choices are the
  secondary model or the differentiable UNet path selected by `use_secondary`.
- Non-finite guidance, clamped score, predicted image or sample values stop the render
  with step, seed, size, precision and attention context. Finite checks synchronize the
  device and their cost is included in corrected benchmark timings.
- Initial noise is now generated on the render device rather than first on CPU. This is
  required for coherent device RNG ownership and intentionally breaks output hashes from
  the pre-correction implementation. Comparisons use the corrected baseline.

## Settings and runtime changes

CLI, benchmark and API inputs pass through one canonical normalizer. It accepts original
Disco JSON, versioned neodisco result JSON and explicit overrides. CLIP checkbox names and
open_clip `(model, pretrained)` pairs round-trip. Validation covers dimensions, model
size, step and skip ranges, eta, finite scales and prompt weights, schedules and their
bounded expansion, cuts at every step, CLIP and prompt presence, batch sizes, precision,
attention and compile modes.

The web runner keeps one active backend and one active CLIP bank. Changing combinations
releases stale references before loading the replacement. Attention selection modifies
only modules owned by one backend, so SDPA cannot contaminate an original-attention run.
The maintained non-reentrant `torch.utils.checkpoint` path replaces the legacy custom
autograd function without changing model keys or architecture; the vendored NOTICE
records this modification.

Effective result JSON uses schema version 1 and stores the resolved seed, actual steps,
device, precision, cut batch, requested/effective compile mode and an init-image path plus
SHA256. A missing init image prevents a portable rerun and is reported as such. The web
API currently accepts one output image per job and rejects `batch_size > 1` before GPU
work because the unchanged UI displays one result.

## Benchmark design

`python -m neodisco.benchmark` has smoke and representative profiles. Manifests include:

- Git revision, dirty state and a package source fingerprint for staged trees without Git
- Python, torch, torchvision and open_clip versions
- CUDA runtime, driver, GPU identity, memory, TF32 and determinism flags
- effective render settings, actual steps and checkpoint SHA256 values
- model loading, compile/first render, warm-up and individual steady render timings
- separate first-render and steady allocated/reserved CUDA peaks
- PNG SHA256 values, a fixed evaluation CLIP distance and paired pixel metrics
- optional named UNet, secondary, total guidance, cutout, CLIP and backward stage totals
  plus a Chrome trace

The manifest is written before expensive work and after every variant. A failure leaves
`status: failed`, includes the error and exits nonzero without a median speed result.
Random seeds are resolved once before variants, so comparisons use paired draws.

Quick smoke:

```bash
python -m neodisco.benchmark --profile smoke --weights weights/disco
```

Representative corrected eager and compiled comparison:

```bash
python -m neodisco.benchmark --profile representative \
  --config examples/cobanov-spaceship.json --weights weights/disco \
  --variant corrected-eager --variant compiled --compile-mode default \
  --runs 3 --warmup 1 --precision bf16 --attention sdpa --cut-batch 64
```

## Validation results

Local CPU environment: macOS, Python 3.14.5, torch 2.14.0, torchvision 0.29.0.

```text
uv run pytest
48 passed, 3 skipped
```

The tests use small real tensors and modules without downloading weights. They cover
schedule bounds, old and saved configs, overrides, API 4xx validation, prompt weight
normalization, fp32 CLIP under outer autocast, nonzero two-tower guidance, LPIPS and chunk
invariance, real cutout gradients, eager/maintained-checkpoint equivalence, instance
attention forward/backward equivalence, toy reference DDIM, RNG restoration and seed
variation including eta/init, both secondary backward paths, finite failures, compile
fallback and OOM propagation, init-state reuse and bounded CLIP caching.

Packaging validation completed locally:

```text
uv sync --frozen --extra dev --extra webui
uv build
fresh wheel install, package imports and CLI help: passed
```

The wheel contains the web HTML/CSS/JavaScript and image assets, the packaged
representative benchmark config, and the vendored LICENSE and modified NOTICE. CI runs
the frozen CPU tests, package build, import smoke and CLI help on Python 3.12. GPU tests
remain explicitly opt-in.

Real CUDA environment: Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
open_clip_torch 3.3.0, NVIDIA RTX 5090 (SM 120), driver 596.21.

The full suite passed there with both real checkpoint integrations and the strict CUDA
mixed-size resize regression enabled:

```text
python -m pytest -c pyproject.toml tests -q --real-weights /path/to/weights
51 passed in 24.40s
```

| Checkpoint | Precision | Secondary | 4-step smoke | Strict repeat |
|---|---|---:|---|---|
| 256 | bf16 | on | finite | exact |
| 256 | bf16 | off | finite | exact |
| 256 | fp32 | on | finite | exact |
| 256 | fp32 | off | finite | exact |
| 512 | bf16 | on | finite | exact |
| 512 | bf16 | off | finite | exact |
| 512 | fp32 | on | finite | exact |
| 512 | fp32 | off | finite | exact |

Ordinary CUDA augmentation repeats restored all RNG states but diverged after four steps.
The observed 256 secondary/bf16 difference reached max absolute `0.356` and RMSE
`0.0102`; fp32 reached `0.098` and `0.00279`. Strict mode produced bitwise identical
results in all eight 256/512 precision/secondary combinations after moving augmentation
to CPU. Every strict check also restored the caller RNG state.
The strict process was started without a preset `CUBLAS_WORKSPACE_CONFIG`; neodisco's
early reference-runtime setup established the policy before model work and the tests
passed.

A real VGG LPIPS init-image run at 320x256, eta 0.8, four steps and one skipped step was
finite. With LPIPS preloaded before each paired render to keep RNG draws aligned, changing
`init_scale` from 0 to 1000 changed the result by RMSE `0.01059`. Reusing the same sampler
for a subsequent render without an init image cleared the prior init and perceptual state.

The isolated historical baseline, before fp32 CLIP correction and normalized prompt
weights, completed the 1280x768 example in `100.720 s`, with `15,589,227,008` bytes peak
allocated and `17,309,892,608` bytes peak reserved. Loading took `18.619 s` and 240 steps
executed. This differs from the README's earlier 143-second observation and is retained
only as historical context. It is not a quality-equivalent baseline.

The corrected eager representative run uses the same 1280x768 example, 250 configured
steps with 10 skipped, three CLIP models, four cutout draws, eta 0.8, bf16 UNet/secondary,
fp32 CLIP, SDPA and cut batch 64:

| Variant | First render | Measured renders | Median | Peak allocated | Peak reserved |
|---|---:|---|---:|---:|---:|
| corrected eager | 133.638 s | 133.736, 132.885, 132.894 s | 132.894 s | 20,918,935,552 B | 23,557,308,416 B |
| compiled default | 192.949 s | 100.488, 100.434, 100.505 s | 100.488 s | 20,910,350,336 B | 22,032,678,912 B |

The compiled steady median is 24.38% lower, or 1.322x the eager throughput. Its first
render costs an additional 59.312 seconds, so the compile cost is recovered during the
third total render, after roughly two subsequent renders. This run used the first full
render as warm-up and set `--warmup 0`; loading took 17.037 seconds for eager and 12.459
seconds for compiled. The corrected 132.894-second eager median is the optimization
baseline; the numerically different historical run is not used to calculate a speedup.

The fixed evaluation CLIP distance was `0.82150` for eager and `0.82902` for compiled.
The paired final images had MAE `0.06032`, RMSE `0.09328`, maximum absolute difference
`0.92941` and PSNR `20.60 dB`. Visual review found coherent blue spaceship, celestial
body and cloud compositions in both, without gross colour collapse or detail failure;
their layouts differed. Ordinary CUDA augmentation is already known to be
nondeterministic, so these trajectory-level pixel differences do not isolate compiler
rounding. Relative to eager run 1, the eager repeat RMSE values were `0.10103` and
`0.08870`; the three compiled pair RMSE values were `0.09328`, `0.09379` and `0.10045`.
The compiled differences therefore fall within the observed fast-mode repeat variability,
which is still not a formal perceptual-equivalence guarantee. Short matched prompt pairs
and strict reference checks remain separate acceptance evidence. The retained manifest is
`docs/benchmarks/rtx5090-20260906-representative.json`; the benchmark's complete ignored
run directory is `benchmark-results/representative-20260906/`.

## Short quality set and profile

Three matched eager/compiled pairs used the 512 checkpoint at 1280x768, bf16, SDPA,
three CLIP models, four cutout draws, eta 0.8 and 50 steps:

| Case | Prompt weights | Eager CLIP distance | Compiled CLIP distance | Pair RMSE |
|---|---|---:|---:|---:|
| spaceship | `a vast blue spaceship orbiting a luminous planet::1` | 0.77795 | 0.77900 | 0.25158 |
| texture | `macro photograph of intricate woven copper and turquoise threads::1` | 0.69288 | 0.73620 | 0.26978 |
| negative | castle and forest `::1.42857`; letters/watermark/text `::-0.42857` | 0.64722 | 0.65993 | 0.27991 |

All six images were visually reviewed. Both modes were coarse, high-contrast and
saturated at 50 steps. The texture case produced large loop-like shapes rather than fine
weave in both modes. No gross collapse, colour failure or detail failure appeared only in
compiled output, but trajectories shifted and the texture case's CLIP distance worsened
by `0.04332`. These short renders are visual sanity checks, not certification of fine
detail or perceptual equivalence. The full 240-step representative pair was substantially
more detailed. Compile therefore remains opt-in and eager remains the default.

A four-step GPU profile attributed about `1.008 s` to UNet work and `1.214 s` to total
guidance. Nested regions reported `0.046 s` secondary, `0.092 s` cutouts, `0.424 s` CLIP
guidance and `0.597 s` guidance-loss backward. Nested labels are not additive. The
guidance-backbone backward region recorded about `25.5 ms` CPU time and zero device time,
showing that profiler event attribution is incomplete for that nested asynchronous region;
it is not evidence that the backward was free.

The sanitized acceptance record, including exact prompts, settings, image hashes, strict
results and profile data, is retained at
`docs/benchmarks/rtx5090-20260906-acceptance.json`. Its six PNGs remain in the ignored
`benchmark-results/final-acceptance-20260906/` run directory.

## Validation limits

The CUDA evidence covers one RTX 5090 software stack. Strict repeatability is scoped to
one supported environment/configuration, and GPU CI is not configured. Ordinary fast-mode
CUDA augmentation is nondeterministic, so matched-seed trajectory differences cannot by
themselves attribute a change to compilation. The short three-prompt set does not replace
full-length perceptual review across a larger corpus. These are evidence limits rather
than blocked implementation work.

## Optimization decisions

Scoped bf16 model forwards, instance-local SDPA, larger cut batches and opt-in UNet
compilation were retained. Broad CLIP autocast was rejected because it changes the
guidance computation. A process-global attention patch was replaced because it invalidates
same-process A/B runs. Compilation was not made the default: it improves repeated steady
renders on the tested stack, but adds a material cold-start cost and the available quality
evidence does not justify silently applying it to every render.

## Checkpoint sources

| File | Source |
|---|---|
| `256x256_diffusion_uncond.pt` | `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt` |
| `512x512_diffusion_uncond_finetune_008100.pt` | `https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt` |
| `secondary_model_imagenet_2.pth` | `https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth` |

The LPIPS domain and sum semantics were checked against the official
`alembics/disco-diffusion` notebook. No model weights or private infrastructure details
are included in repository artifacts.

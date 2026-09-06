# Disco compatibility implementation

Date: 2026-09-07. This supersedes the original-equivalence interpretation of the
2026-09-06 modernization acceptance, not its historical measurements.

The supported target is the original still-image DDIM guidance algorithm on a modern
runtime. The historical `TimeToDisco(0)_14.png` is not a golden reproduction target:
its notebook version and batch/RNG position are unknown.

## Implemented behavior

- Each CLIP model receives independent draws at its native cut resolution, in original
  model-outer/draw-inner RNG order. Only encoder evaluation of that model's cuts is
  batched. No cuts or activations are shared between CLIP models.
- Dango cutouts use vendored ResizeRight, pinned at `510d4d5`, with its MIT licence.
  An owner-scoped interpolation-plan LRU reuses only constant fields/weights/padding,
  bounded to 512 entries and 32 MiB. It retains no images or autograd graphs. The
  original dimension ordering and weight-application operations remain unchanged.
- Pixel guidance no longer invents an inverse path from the blend to the prediction.
  The imported range scale stays in the config; its gradient with respect to the blend
  is zero, as in the notebook. Generic standalone image/latent guidance keeps its
  explicit image-space regularizer.
- Secondary forwards always run in fp32, including under enclosing AMP. CLIP remains
  fp32 for modern stability; UNet bf16/fp32 and attention/compile remain selectable.
- Base diffusion steps are `(1000//steps)*steps`, with DDIM striding and scaled model
  timesteps. Cut schedules index `999-int(original_scaled_t)`. Scalar and scheduled
  `cut_ic_pow` pass through original-config import, CLI, API, and the sampler.
- Skipping without an init image noises a zero image to the starting timestep. Preview
  and saved images use the unconditioned clean prediction, as in the kostarion fork;
  the DDIM update still uses score conditioning.
- Original `skip_augs` is imported. PLMS, enabled Perlin/fuzzy generation and image
  prompt guidance are rejected rather than silently rendered with another algorithm.
- The original cond_fn's NaN image-gradient recovery is restored: only that step's
  guidance becomes zero, with a warning and `guidance_nan_steps` in the result record.
  Invalid model/secondary predictions, infinite gradients and non-finite diffusion
  samples still fail. Generic non-pixel guidance remains strict. Tests verify the
  source recovery branch and that it cannot mask a non-finite UNet.
- Sidecars identify `sampling_semantics=disco-2026-09-07`, resize/draw/range semantics,
  secondary precision and the one-render seed scope. Existing saved images remain intact.
- The web example stays at 250 configured steps, skip10, **240 actual iterations**.

## Verification boundaries

`tests/reference/disco.py` holds source-extracted Dango, cond_fn and loss functions
from alembics/disco-diffusion `37eb39b`. Golden cutout tensors and input gradients
were generated with an independent original ResizeRight checkout. Tests cover:

- augmented and unaugmented cutout pixels/gradients;
- original cond_fn scores and post-call RNG state with independent mixed-resolution
  models and different encoder chunk sizes;
- complete tiny sampling loops with and without secondary guidance;
- non-divisor timestep maps, rescaled model times and source schedule indices;
- skipped zero-init, guided DDIM updates and unconditioned output selection;
- cache eviction, byte limits, graph isolation and inference-to-autograd reuse;
- original-config imports and effective settings provenance.

These oracles use toy encoders/backbones to isolate mathematics. Real 256/512 loading,
finite CUDA generation, deterministic resize backward and a complete API job with
result metadata have separate integration tests.
No claim is made that fp32 OpenCLIP + bf16 UNet equals historical fp16 OpenAI CLIP +
fp16 UNet numerically. Strict determinism refers to one modern stack. Per-job seeds do
not emulate later images in an old notebook's continuous batch RNG stream.

## GPU measurements and release

Final timings and release status are added after full-resolution validation.
Verification totals 82 tests on the RTX 5090 host: the 81-test suite (78 CPU/oracle
+ 3 CUDA integration) and the subsequently added real API integration test all pass.
Local CPU tests and wheel/sdist builds also pass.
An uncached eager baseline completed in 223.366 seconds. Its subsequent compiled
render exposed a non-finite guidance gradient. This failed run is not a successful
speed or quality result and does not prove that compile caused the NaN. The reference
NaN image-gradient recovery had been omitted from the port; that behavior is now
restored and recorded instead of silently failing or changing the loss formula.
A subsequent cached compile diagnostic render completed in 147.098 seconds without
triggering recovery. The precise operation producing the earlier NaN was not recovered
from that failed process. CUDA stochastic-gradient nondeterminism remains a limitation.


### Final cached path

| Condition | Seconds | Peak allocated VRAM | NaN guidance steps |
|---|---:|---:|---:|
| Eager, seed 1290357248 | 170.695 | 13.283 GiB | 0 |
| Warm compile, same seed | 137.591 | 13.283 GiB | 0 |
| Warm compile, seed 7 | 137.616 | 13.284 GiB | 0 |

Plan caching shortened the eager run by 23.58%. Warm compile
shortened the cached run by 19.39% (33.10 seconds). Each condition has one
render; these are not medians. Compile setup/warmup is excluded, and an existing disk
compiler cache was present. First-use latency is not established by these timings.
The cache held 512 plans using about 12.3 MiB of tensors, within its 32 MiB bound.

Visual inspection: both same-seed outputs preserve the broad cloud/spaceship/blue-orb
composition, but local geometry, highlights and painted texture differ between eager
and compiled renders. Neither reproduces the historical image. Ordinary CUDA backward
nondeterminism and floating-point differences prevent attribution to compilation alone.
Speed is measured; perceptual or pixel equivalence is not claimed.

Evidence: [machine-readable manifest](benchmarks/rtx5090-20260907-disco-compatibility.json).
Local PNGs and scripts: `benchmark-results/fidelity-final-20260907/`.
The release on the user's RTX 5090 host enables `--compile default`, with two compile
workers. Package/CLI defaults remain eager. The existing output directory and service
ports stay the same. Deployment rollback restores the saved previous systemd drop-in.

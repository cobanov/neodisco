# Neodisco modernization: implementation and acceptance plan

Date: 2026-09-06. Baseline: `03d5792d75c7d68adab90b414047b4bedb6e99e7`.
Status: implemented by GPT-5.6 Sol (high reasoning); coordinator review, real
RTX 5090 validation and visual review complete. See the implementation report for
measured results and the limits of the short three-prompt quality set.

## Objective and scope

Run the original Disco Diffusion checkpoints reliably and efficiently on modern
NVIDIA CUDA GPUs while preserving the CLIP-guided Disco mechanism. Deliver working
code, meaningful regression tests, a reproducible benchmark, measured GPU results,
and accurate installation/use documentation. A faster render with a different
guidance workload is not an equivalent optimization.

Keep the existing CLI, Python package and web application usable. Preserve the
default Disco `clamp_max` path, checkpoint architecture and vendored MIT notices.
Animation, model training, a replacement latent model, new visual design, AMD/MPS
production support, distributed inference and public deployment are outside this
iteration. CPU execution is useful for small tests; do not promise CPU render speed.

The user approved Sol to implement the complete plan. The coordinating agent owns
environment discovery, numerical review, final acceptance and the Obsidian log.
Sol owns application code, tests, benchmark tooling, CI and repository docs. Work
in this checkout, preserve unrelated changes, and do not push or replace a running
deployment as part of the implementation. Use an isolated GPU staging directory.

## Evidence and known gaps

The baseline README records 201 s / 14.5 GB for original fp32 attention and 143 s /
20.8 GB for SDPA + bf16 + cut batch 64 on an RTX 5090 at 1280x768, 250 steps, three
CLIP models and four draws. There is no checked-in benchmark or test suite. Treat
these as historical observations, not reproduced results or a guaranteed baseline.

The initial code review found these paths; reproduce relevant issues in tests
before changing them:

1. `PixelBackend.sample` wraps `ddim_sample` in autocast. Its nested `cond_fn` reaches
   CLIP under that same context; fp32 weights/input casts do not disable AMP.
2. Only initial noise uses the seed-specific CPU generator. Cutouts, augmentations
   and DDIM's `randn_like` use unrelated global RNG state.
3. `guidance_strength` and `through_model` are accepted but not read.
4. `set_init` loads LPIPS, but no perceptual loss contributes to the gradient.
5. The README's non-finite-sample guard is absent from the current sampler.
6. CLI/device/autocast defaults and server defaults differ. CUDA/bf16 capability
   checks are absent; web cut batch is hardcoded to 64.
7. The web CLIP bank cache retains every model combination on GPU. The fast-attention
   monkeypatch is process-global and cannot restore the original path for an A/B run.
8. Imported Disco CLIP tuples do not match the web runner's checkbox-name format.
   Result JSON and init-image metadata also need a coherent reload path.
9. Legacy custom checkpointing needs explicit AMP/backward coverage, particularly
   when secondary guidance is disabled. Avoid fixing one path while breaking it.
10. Dependencies are broad/unlocked and current documentation overstates fidelity,
    supported configurations and the effect of some options.

## Execution order

Each phase ends with its tests and a short result in `docs/implementation-report.md`.
Separate correctness fixes from performance experiments in the report. Do not label
GPU-dependent acceptance complete merely because CPU tests pass.

### 1. Establish a test and measurement harness

- Add a development extra with pytest and web API test dependencies; tests must not
  download weights, invoke network APIs or allocate the full production UNet.
- Use small real tensors/models for gradients and sampler algebra. Mock heavyweight
  loading only at the boundaries (CLIP downloads, LPIPS weights, checkpoint loading).
- Add an opt-in GPU/real-weights integration marker with explicit skip reasons.
- Build a benchmark CLI that records the revision and dirty state, Python/torch/
  torchvision/open_clip versions, CUDA/driver/GPU identity, effective settings,
  precision, attention/compile mode, checkpoint SHA256 and image hashes.
- Separate loading time, compile/first-render time, warm-up and steady-state render
  timing. Synchronize CUDA at timing boundaries and reset peak memory statistics.
  Report allocated and reserved peaks, median and individual measured times.
- Save PNGs and JSON results into an ignored output directory. Do not embed private
  hostnames, absolute personal paths, secrets or huge weights in public artifacts.
- Support a quick smoke case and a representative 250-step case, paired seeds and
  eager/compiled or original/SDPA variants. Do not silently reduce cuts/CLIPs/steps.

Acceptance: one command produces a machine-readable run manifest and renders;
CPU unit tests collect and run without real model downloads; failures produce a
non-success result rather than a speed figure.

### 2. Repair numerical correctness and reproducibility

**Precision boundaries**

- Run CLIP text/image scoring and guidance loss accumulation explicitly outside
  autocast in fp32. Scope UNet/secondary autocast to their forward calls, not the
  complete sampler update or entire guidance/backward pass.
- Match the autocast device type to the selected device. Select bf16 only where
  supported; provide explicit fp32 fallback. Reject incompatible forced precision
  combinations early. Keep fp16 opt-in and describe its limits.
- Test real operation/output dtypes under an outer autocast context, not just the
  dtype of weights. Cover secondary-on and secondary-off backward paths.
- If replacing legacy custom checkpointing, use the maintained torch checkpoint
  API with behavior tested against the eager path, preserving checkpoint weights
  and architecture. Do not apply inference_mode to tensors needed by guidance.

**RNG ownership**

- Seed every stochastic source used during one render, including initial noise,
  crop geometry, augmentations and stochastic DDIM updates. A scoped RNG context
  using `fork_rng` and restoration is acceptable with the serialized runner; an
  explicit-generator design is also acceptable if all torchvision operations are
  covered. Preserve the same random draws between benchmark variants.
- Restore caller RNG state; make the serialization assumption explicit. Avoid
  concurrent use of global RNG contexts. Include init-image and eta > 0 tests.
- Promise reproducibility within the same supported environment/configuration,
  not bitwise equality across GPU architectures or torch releases. Provide a
  deterministic test/reference mode where needed; record its performance impact.

**Guidance semantics**

- Preserve `clamp_max > 0` as the default Disco score-gradient behavior.
- Resolve the legacy `--strength` option explicitly: for `clamp_max == 0`, specify
  the step-relative formula and test that strength 0 gives unguided sampling and
  positive strengths affect the score as documented. Inspect the earlier backend
  history for the intended formula, but do not copy incompatible sampler algebra.
  If a stable formulation cannot be justified, reject/deprecate the option clearly
  and document that decision instead of inventing an unverified sampler. Never
  silently accept a no-op flag.
- Implement LPIPS init-image loss in the intended image domain, retaining the
  differentiable path and applying it once per guidance pass, independently of
  cutout chunk size and draw count. Check original Disco source if uncertain about
  the domain/normalization. Init-image noise initialization and skip_steps continue
  to work even when perceptual guidance is zero.
- An explicitly requested nonzero init_scale with missing LPIPS should have a clear
  actionable failure; do not claim successful use while silently ignoring it.
- Retire or implement `through_model` with explicit compatibility handling; avoid
  changing exported guidance methods used by the separate latent project without
  documenting the contract. Check consistency of `loss` and `image_gradient`.
- Reject non-finite gradients/samples before saving a successful PNG. Include a
  useful step/settings context. Do not turn NaNs into a black image. Measure any
  additional per-step GPU synchronization introduced by these checks.

Acceptance: same seed reproduces a render and restores caller RNG; different seeds
change output; default Disco score math matches the vendored reference; init_scale
changes the actual gradient; cut-batch partitioning preserves the loss/gradient
within a justified tolerance; finite failures surface to both CLI and web.

### 3. Normalize settings and resource management

- Share canonical settings/validation between CLI and server without replacing the
  frontend. Parse original Disco JSON, internal saved JSON and explicit overrides
  consistently, including CLIP identifiers and prompt weights.
- Validate positive dimensions/multiples of 64, checkpoint sizes, steps 1..1000,
  `0 <= skip_steps < steps`, eta bounds, finite scales, valid schedules/cut counts,
  at least one cut/CLIP/prompt, valid batch values and seed handling. Reject invalid
  inputs before heavyweight loading. Bound the schedule expression work to prevent
  tiny inputs expanding into unbounded lists; continue avoiding eval.
- Introduce explicit device/precision/cut-batch options and an `auto` memory profile
  based on actual available memory with conservative headroom. Honor manual values;
  record what was actually selected. Calibrate rather than claiming all resolutions
  fit on a given VRAM size. If implementing OOM retry, restart the whole render with
  the same seed and smaller chunks; never continue a half-completed step.
- Keep at most a bounded number of CLIP banks (prefer one active bank or shared
  immutable towers). Release stale GPU references; empty_cache alone is not eviction.
- Make original-vs-SDPA attention instance-specific or reversibly scoped, so creating
  an SDPA backend does not contaminate a later baseline backend.
- Save versioned effective settings sufficient to rerun: resolved seed, actual
  precision/batch/compile mode, supported sampling options and a durable init-image
  reference/hash. Do not claim portable reruns when the required image is absent.
- Restore existing result JSON/history compatibly; report input errors as API 4xx.
  Preserve preview, queue, history and existing UI behavior.

Acceptance: example Disco JSON works through CLI and API; result settings roundtrip;
changing CLIP combinations does not grow GPU residency unboundedly; invalid settings
fail before GPU work; explicit precision and memory options are respected.

### 4. Optimize against the corrected baseline

- Re-measure after phase 2; this is the new quality-equivalent baseline. Keep the
  old 143-second result separately labeled, because fp32 correction can change cost.
- Add opt-in UNet `torch.compile`. Measure supported modes, graph breaks and shape
  recompilation. Compile only a stable forward initially, not random cutout Python
  or the complete sampling loop. Keep resolution/device/precision in cache identity.
- Lazy compile failures must surface clearly or fall back to eager with the effective
  mode recorded. Never catch unrelated OOM/numerical errors and relabel them compile
  fallback. Test real eager fallback using a controlled failing compiler.
- Profile secondary/CLIP/cutout stages after UNet improvements. Only keep additional
  optimizations that the profile justifies. Evaluate launch overhead and tensor
  layout; preserve fp32 CLIP scoring and random crop semantics.
- Keep a force-eager/reference mode. Do not enable compile by default without an
  actual end-to-end benefit for this workload including the user's first-run cost.
- Compare matched gradients and images, plus a fixed evaluation CLIP distance and
  perceptual/image metrics if available. CLIP distance alone does not certify Disco
  fidelity. Save paired images for visual inspection; document chosen tolerances.

Performance hypothesis, not a release requirement: 15–30% less steady-state time
than the corrected eager baseline; 30–40% is a stretch. No 2–3x promise. Lower-VRAM
profiles optimize successful completion and may be slower. Do not add stage-specific
speedup percentages as if they were independent end-to-end gains.

Acceptance: matched-workload benchmark, measured first-run/steady-state tradeoff,
no unexplained quality shift, eager escape hatch, and honest reporting even if an
experiment fails to improve speed.

### 5. Package, document and finish

- Create a reproducible tested environment (lockfile or explicit constraints) while
  keeping library installation portable. Do not pin every platform to a CUDA wheel
  index. State actual tested Python/torch/CUDA combinations and distinguish minimum
  declared dependencies from verified support. Keep optional LPIPS/web dependencies.
- Add CI for CPU regression tests and package build/import. GPU tests remain opt-in
  unless GPU CI is actually configured; no green badge implying GPU coverage.
- Update README claims, defaults, install commands, benchmark instructions, known
  limits, strength/init semantics and compatibility table. Keep design/assets and
  attribution. Use the readme-schema skill when editing README.
- Add `docs/implementation-report.md` with phase status, commands, test totals/skips,
  benchmark tables, artifact locations, quality observations, rejected experiments
  and any external blockers. Keep public reports free of private infrastructure data.
- Coordinator updates the existing Obsidian neodisco note under latent-model; do
  not open a duplicate project or edit unrelated historical diary entries.

## Required validation matrix

| Layer | Cases |
|---|---|
| CPU tests | schedule parsing and bounds; old/internal config and overrides; API validation; attention forward/backward equivalence; toy DDIM reference; RNG restoration/repetition; nonfinite failures; LPIPS gradient using a differentiable stand-in; chunk invariance; compile fallback; bounded cache |
| CUDA smoke | real 256 and 512 checkpoints; secondary on/off; bf16/fp32; same seed repeated with eta 0.8; init-image + LPIPS; eager/compiled; rectangular frame |
| Representative run | 512 model at 1280x768, 250 steps, three CLIPs, four draws, eta 0.8, example settings; corrected eager vs optimized; at least 3 measured runs per selected final variant following warm-up |
| Quality set | at least 3 prompts/seeds including spaceship, fine texture and a negative-weight prompt; matched settings and saved paired images; metrics plus visual review |
| Packaging | fresh environment install, CLI help, web route smoke, tests, wheel includes web assets |

Run a small real-model smoke before the expensive matrix. Reuse downloaded weights
read-only. If hardware/time prevents some runs, preserve exact runnable commands and
mark them pending, not passed. Tested on one GPU does not imply tested on all GPUs.

## Handoff completion checklist

- [x] Plan saved and Sol implementation started.
- [x] Numerical correctness and RNG regressions fixed and tested.
- [x] Settings/API/memory/attention isolation implemented and tested.
- [x] Benchmark and optional compile path implemented and exercised.
- [x] Real GPU evidence collected, or precise external blocker recorded.
- [x] Installation/CI/README and implementation report complete.
- [x] Coordinator review findings resolved and affected tests rerun.
- [x] Obsidian project log updated; final user report distinguishes measured results
  from estimates and pending GPU coverage.

Acceptance evidence: 48 offline CPU tests; 51 tests on the RTX 5090 including three
integration tests; strict repeats across both real checkpoints, both precisions and
both secondary modes; real VGG LPIPS and rectangular init rendering; three steady
runs per representative variant. Compile reduced steady time from 132.894 s to
100.488 s (24.38%), with a 192.949 s first render. The three-prompt visual set used
50-step matched pairs and remains a sanity check, not a perceptual-equivalence
guarantee. Compile stays opt-in. Raw manifests are retained under `docs/benchmarks/`.

## Reviewed implementation decisions

- 2026-09-06: reject the explicit legacy `--strength` option with a migration
  message. Preserve raw Disco score guidance when `clamp_max == 0`; do not copy
  the old step-relative formula from the incorrect hand-written sampler.
- 2026-09-06: LPIPS applies to blended `x_in` and init, in [-1, 1], as
  `lpips(x_in, init).sum() * init_scale`, once per guidance pass. Verified against
  the original `alembics/disco-diffusion` notebook.
- 2026-09-06: RTX 5090 (32 GB), driver 596.21, Python 3.12, torch 2.11.0+cu128,
  torchvision 0.26.0+cu128 and open_clip_torch 3.3.0 are available for isolated GPU
  validation. Existing deployment and its uncommitted checkout remain untouched.

## Instruction to the implementing Sol agent

Read this entire plan and applicable repository instructions. Implement every phase
in order, using actual tests and documenting decisions. Do not stop after scaffolding
or a plan-only response. The coordinator is preparing remote GPU validation; send
test/setup requirements and a concise checkpoint when the corrected sampler is
ready. Work locally on code and CPU tests; avoid racing the coordinator's remote
GPU jobs. Do not spawn further agents. Treat unavailable hardware as a validation
limitation, not permission to skip independent code/test/documentation work. Request
coordinator review for the strength formula and any gradient-domain changes. Finish
with a report of changed files, tests, measurements and remaining blockers.

<p align="center">
  <img src="examples/cobanov-spaceship-1280x768.png" alt="A blue-lit capital ship above painterly storm clouds, rendered from a 2022 Disco Diffusion settings file" width="700">
</p>

<p align="center">
  Disco Diffusion's original checkpoints and CLIP guidance on current PyTorch and NVIDIA GPUs.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-d6249f?labelColor=1a1a1a">
  <img alt="CPU tests" src="https://img.shields.io/badge/CPU%20tests-48-d6249f?labelColor=1a1a1a">
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-MIT-d6249f?labelColor=1a1a1a"></a>
</p>

---

Disco Diffusion produced dense, fragmented images by making an unconditional ImageNet
diffusion model fight a text signal assembled from many random CLIP crops. The models
still work, but the notebook depended on a 2021 Python and PyTorch stack and its runtime
assumptions do not hold on recent CUDA hardware.

neodisco loads those original checkpoints and the settings JSON files left by the
notebook. It keeps the Disco score gradient, cutout schedules, secondary model and init
image path while putting precision, randomness and failures behind explicit controls.

- **Original checkpoints.** It supports OpenAI's 256 model, Katherine Crowson's 512
  finetune and the secondary model without converting their state dictionaries.
- **Original settings import.** Prompt weights, CLIP checkbox names, schedules, eta,
  skip steps, seed and init settings pass through one validated configuration layer.
- **Bounded modern execution.** CLIP scoring stays in fp32, UNet AMP is scoped to model
  forwards, attention is selected per backend, and the web server retains one CLIP bank.
- **Reproducible reference mode.** One render owns and restores its RNG state. Strict
  mode moves stochastic cutout augmentation to CPU to avoid nondeterministic CUDA
  interpolation backward operations.
- **Measured failures.** Non-finite gradients or samples stop the render with step and
  runtime context instead of producing a successful black PNG.

## Install

Python 3.10 or newer is declared. The tested environments are listed in the
[implementation report](docs/implementation-report.md). Install PyTorch for your CUDA
platform first if its standard wheel is not suitable, then install neodisco:

```bash
pip install -e .
pip install -e ".[init,webui]"   # LPIPS init guidance and the web application
```

For development, `uv sync --frozen --extra dev --extra webui` enforces the checked-in lockfile.
The vendored sampling subset from OpenAI guided-diffusion keeps its MIT licence and
notice under `neodisco/backends/_guided_diffusion`.

| Environment | Validation |
|---|---|
| Python 3.12, torch 2.11 + CUDA 12.8, RTX 5090 | real 256/512 checkpoint smoke |
| Python 3.14, torch 2.14, macOS CPU | offline regression tests |
| Python 3.10+ and torch 2.4+ | declared compatibility, not fully measured |

Download the 256 or 512 diffusion checkpoint and, for the default guidance path,
`secondary_model_imagenet_2.pth` into `weights/disco`. The source URLs and expected
filenames are recorded in [the implementation report](docs/implementation-report.md).

## Use

Run an original settings file:

```bash
neodisco --disco-config examples/cobanov-spaceship.json --out spaceship.png
```

Or provide weighted prompts directly:

```bash
neodisco "an ukiyo-e city::1" "neon::-0.2" \
  --image-size 512 --width 1280 --height 768 --steps 250 --eta 0.8 \
  --precision auto --attention sdpa --cut-batch auto --out city.png
```

Every successful CLI render writes a PNG and a versioned JSON record beside it. The
record contains the resolved seed, precision, cut batch, attention and compile mode,
plus an init-image hash when used. Input dimensions must be positive multiples of 64.

`clamp_max` remains Disco's supported guidance control. `0.05` is the default; zero
keeps the raw, unclamped Disco score. The former `--strength` option is rejected because
its step-relative formula was tied to an incorrect hand-written sampler update. It was
a no-op in the previous reference-DDIM path, so accepting it would be misleading.

An init image controls composition through `--skip-steps`. A nonzero `--init-scale`
adds `LPIPS(x_in, init).sum()` in Disco's `[-1, 1]` blended-image domain and requires the
`init` extra. Missing LPIPS is an error. Example:

```bash
neodisco --disco-config settings.json --init-image small.png \
  --skip-steps 125 --init-scale 1000 --out large.png
```

Run the existing web interface with the same validation and runtime policy:

```bash
neodisco-web --weights weights/disco --out outputs \
  --device auto --precision auto --attention sdpa --cut-batch auto
```

The UI remains at `http://127.0.0.1:7870`. The server serializes GPU jobs, reports input
errors as HTTP 400 responses, preserves previews and history, and restores both original
Disco JSON and versioned result JSON. Start it with `--deterministic` when the web process
must accept strict reference jobs; this prepares CUDA before model loading.

## Benchmark

The smoke profile runs the real 256 checkpoint without reducing work silently. The
representative profile includes a packaged 1280x768, 250-step config with three CLIP
models, four cutout draws and eta 0.8:

```bash
neodisco-benchmark --profile smoke --weights weights/disco

neodisco-benchmark --profile representative \
  --config examples/cobanov-spaceship.json --weights weights/disco \
  --variant corrected-eager --variant compiled --compile-mode default \
  --runs 3 --warmup 1
```

On the tested RTX 5090 stack, compiled steady renders measured 100.488 seconds versus
132.894 seconds eager, a 24.38% reduction. Compilation added 59.312 seconds to the first
render and paid back during the third total render. See the
[implementation report](docs/implementation-report.md) for settings, memory and quality
limits.

Each run directory contains PNGs and a machine-readable manifest with code provenance,
package and GPU versions, checkpoint hashes, effective settings, actual executed steps,
load and render timings, allocated and reserved peaks, CLIP distance and paired image
metrics. Add `--profile-stages` for a Chrome trace. A failed render writes a failed
manifest and exits nonzero, without publishing a speed figure.

## How it is built

CLIP image and text operations explicitly disable outer autocast. bf16 and fp16 are
accepted only on compatible CUDA devices; fp16 remains opt-in because wide attention can
overflow. `torch.compile` targets the stable UNet forward and defaults to eager. Compiler
failures may fall back to eager and are recorded; OOM and numerical failures propagate.

Default CUDA augmentation can differ between repeated runs even with identical random
draws because interpolation backward is nondeterministic. `--deterministic` preserves
the transforms but computes them on CPU, which is slower. Reproducibility is promised
within one supported software and hardware configuration, not across PyTorch releases or
GPU architectures.

CPU rendering is useful for small tests, but full checkpoint rendering is intended for
CUDA. Animation, training, replacement latent models, AMD/MPS production support and
distributed inference are outside this release.

## Licence

MIT. Vendored guided-diffusion code is MIT, copyright OpenAI.

<p align="center">
  <img src="examples/cobanov-spaceship-1280x768.png" alt="A vast blue-lit capital ship above painterly storm clouds, a laser column behind it and small fighters below: a 2022 Disco Diffusion settings file rendered today" width="900">
</p>

<p align="center">
  Disco Diffusion's models and technique, running on current GPUs.<br>
  The 2021 checkpoints, PyTorch 2.11, a card made this year.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-d6249f?labelColor=1a1a1a">
  <img alt="torch" src="https://img.shields.io/badge/torch-2.4%2B-d6249f?labelColor=1a1a1a">
  <img alt="checkpoints" src="https://img.shields.io/badge/checkpoints-still%20up-d6249f?labelColor=1a1a1a">
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-MIT-d6249f?labelColor=1a1a1a"></a>
</p>

---

Disco Diffusion produced a look that nothing since has reproduced: over-detailed,
fractal, dreamlike, composed of fragments that half-belong together. Then the
notebook rotted. Its code targets PyTorch 1.10 and Python 3.7, and it will not run
on a card made after 2022.

The models themselves are fine. They are still downloadable, they still work, and
this repository runs them.

The image above is OpenAI's 256x256 unconditional ImageNet model from July 2021, on
an RTX 5090 under PyTorch 2.11. Guidance strength 0.5, 2.0, 8.0, left to right.
Prompt: *a fractal cathedral of glowing coral, intricate, dreamlike*.

- **The original checkpoints, not a lookalike.** Both unconditional ImageNet
  diffusion models Disco used, loaded as they are.
- **Disco settings files load directly.** Point `--disco-config` at a `.json` from
  the notebook and the prompts, weights, cutout schedules, `eta`, `skip_steps` and
  seed all come through.
- **Any frame size that is a multiple of 64.** The network is convolutional and its
  attention runs on whatever grid it is handed, which is how Disco made widescreen
  frames from a model trained at 512x512.
- **Guidance that survives a change of sampler.** The gradient is normalised against
  the step's own magnitude, so a strength number means the same thing whatever the
  sampler underneath.

## Models

Both are unconditional ImageNet diffusion models, and both are still up:

| Checkpoint | Source | Size |
|---|---|---|
| `256x256_diffusion_uncond.pt` | OpenAI, July 2021 | 2.21 GB |
| `512x512_diffusion_uncond_finetune_008100.pt` | Katherine Crowson's finetune, the one Disco used by default | 2.23 GB |
| `secondary_model_imagenet_2.pth` | Crowson's secondary model; Disco took the CLIP gradient through this by default | 53 MB |

```bash
mkdir -p weights/disco && cd weights/disco
curl -LO https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
curl -LO https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt
curl -LO https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth
```

The secondary model matters more than its size suggests. Disco's `use_secondary_model` was
on by default, and it changes the *direction* of guidance, not just its cost: the CLIP
gradient is taken through a 14M-parameter network that predicts the clean image, rather
than through the 550M-parameter UNet, and the result is the softer, more painterly steer
people remember. On the same settings file, CLIP distance to the prompt after 250 steps
drops from 6.10 (unguided) to 5.25 with the secondary model, against 5.77 through the
UNet. Pass `--no-secondary` to use the UNet path.

## Install

```bash
pip install -e .
```

The sampling half of [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
is vendored under `neodisco/backends/_guided_diffusion` (MIT, see the NOTICE there).
The upstream package depends on `mpi4py` and `blobfile` for training and data loading,
neither of which sampling needs and both of which make it awkward to install today.

## Use

If you have a settings file from the Disco notebook, point it at that. Animation and
video keys are ignored; everything else comes through.

```bash
python -m neodisco.cli --disco-config examples/cobanov-spaceship.json --out spaceship.png
```

![a Disco settings file from 2022, run today](examples/cobanov-spaceship-1280x768.png)

That is `examples/cobanov-spaceship.json`, a settings file from 2022, run unchanged:
Crowson's 512 model at 1280x768, three CLIP backbones, Disco's cutout schedules,
`eta 0.8`, the original seed. *An enormous sci-fi spaceship attacking a massive
deathstar in front of a black hole, by Greg Rutkowski and Thomas Kinkade; blue color
scheme.*

Any flag given on the command line overrides the file, so a quick low-resolution
preview of an old config is:

```bash
python -m neodisco.cli --disco-config settings.json --width 640 --height 384 --steps 60
```

Without a file, describe the picture directly:

```bash
python -m neodisco.cli \
  "a lush alien jungle city at golden hour, by Moebius" \
  --image-size 512 --width 1280 --height 768 --steps 250 --eta 0.8 \
  --overview-cuts "[12]*400+[4]*600" --inner-cuts "[4]*400+[12]*600" \
  --cutn-batches 4 --strength 3.0 --out out.png
```

Weight several prompts against each other with `::`:

```bash
python -m neodisco.cli "an ukiyo-e woodblock print::1.0" "neon::0.4" --image-size 256
```

### The knobs that change the look

| Flag | Does |
|---|---|
| `--strength` | how hard the prompt pulls against the prior |
| `--eta` | 0 is deterministic DDIM, 1 is ancestral DDPM. Disco used 0.8 |
| `--cutn-batches` | how many independent cutout draws are averaged per step |
| `--overview-cuts` | how much the prompt affects overall composition. A number, or a Disco schedule string like `"[12]*400+[4]*600"` |
| `--inner-cuts` | detail density. Low is calm, high is the full fractal surface |
| `--steps` | how long the prior and the guidance are allowed to fight |
| `--tv-scale` | smooths the speckle that guidance introduces. 0 is grittier |
| `--clip-models` | more backbones means more compositing, less literal |
| `--cut-batch` | memory only. Lower it if the card runs out |
| `--grad-checkpoint` | trades speed for memory inside the UNet |
| `--fp16` | half-precision UNet; opt-in, see the notes below |

## Why nothing else looks like this

The Disco look was never a style you could ask a model for. It was the visible
residue of a mechanism:

- an **unconditional** diffusion model trained on ImageNet, which knew textures but
  was never taught to compose a scene and never saw a word of text;
- **CLIP guidance** applied through dozens of random crops at random scales, so
  detail accumulated independently at every scale with nothing tying it together;
- **hundreds of steps** at high guidance, long enough for the two to grind against
  each other.

Current text-to-image models cannot make these pictures because they are too good.
They have strong text conditioning and learned composition, so they produce coherent
images. The Disco look is what a weak prior fighting a weak guidance signal looks
like, and the only way to get it back is to run that fight again.

## How it is built

Four things go wrong in ways that are quiet rather than loud, and all four cost time.

**An absolute guidance scale does not survive the sampler.** Disco's `clamp_max=0.05`
is tuned for its own step parameterisation. Carry the number across to a different
sampler and the guidance term, once multiplied by the step size, rounds away to
nothing: the image comes out clean, coherent and completely ignoring the prompt. Here
the gradient is normalised and scaled relative to the step's own magnitude, so
`--strength 1.0` means the prompt pulls about as hard as the prior does, whatever the
sampler.

**The clean-image estimate should not be rebuilt from the noise prediction.** Writing
`eps = (x - sqrt(a) * x0) / sqrt(1 - a)` and then differentiating through it is the
obvious formulation and it divides by zero at the end of sampling. Take the gradient
with respect to the estimate directly and divide by `sqrt(alpha_bar)` instead.

**Cut and augment in [0, 1], not [-1, 1].** ColorJitter and the small additive noise in
Disco's augmentation stack assume a [0, 1] image. Run them on [-1, 1] data and the colour
augmentations misbehave, which quietly shifts the palette of every run.

**Use Disco's `ddim<N>` respacing.** `space_timesteps(1000, "ddim250")` picks every
fourth timestep from zero; the plain `"250"` spacing is a different set of timesteps.

**fp16 can overflow on large frames.** The checkpoints were trained in fp16 and run
fine that way at 512x512, but at widescreen sizes the attention over a few thousand
tokens overflows on some seeds and the frame comes out blank. The sampler now stops
with a clear error instead of writing the blank. The UNet therefore runs in fp32 by
default; `--fp16` is opt-in for square frames on small cards.

**Group cutout gradients in image space, not through the decoder.** Chunking the CLIP
pass to save memory is necessary, but if each chunk backpropagates all the way through
a decoder, sampling slows by the number of chunks for no change in the answer.

The same guidance also runs against our own rectified-flow latent models, trained from scratch on a single GPU; that lives in a separate project, `cobanov-diffusion`, which imports this package for the cutouts, losses and CLIP bank.

## Licence

MIT. Vendored guided-diffusion code is MIT, copyright OpenAI.

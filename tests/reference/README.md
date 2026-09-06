# Original Disco test oracles

`disco.py` contains AST-selected definitions (only trailing whitespace removed) from
alembics/disco-diffusion `37eb39bfe0e7310c86c244859b789a5346754251`:
MakeCutoutsDango, cond_fn, spherical_dist_loss, tv_loss, range_loss.
Notebook globals are injected by tests. Licence: LICENSE.disco.

`cutouts.npz` was generated from that Dango definition and the independent
assafshocher/ResizeRight checkout `510d4d5b67dccf4efdee9f311ed42609a71f17c5`,
not neodisco's resize implementation. Input: CPU float32 torch.linspace(-1.3,1.2,
3*40*64), shaped 1x3x40x64; RNG seed 123 reset per case; output size16;
overview4/12, inner3, power1, grey0.2; augmentations off/on. Stores cut pixels
and the input gradient of sum(cuts**2). See regenerate_cutouts.py.

Tests compare numerical values with explicit tolerances, original cond_fn scores,
post-call RNG states, native mixed CLIP resolutions and complete small sampling loops.
The encoders/secondary/UNet in those CPU tests are small deterministic substitutes;
real model loading and finite CUDA output are covered separately by integration tests.
These tests do not assert reproduction of a historical notebook batch or CUDA bit identity.

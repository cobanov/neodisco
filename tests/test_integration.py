from pathlib import Path

import pytest
import torch
import torch.nn as nn

from neodisco.cli import render
from neodisco.clip_bank import ClipBank
from neodisco.runtime import prepare_reference_runtime, render_rng
from neodisco.settings import DEFAULTS, normalise_settings


@pytest.mark.integration
@pytest.mark.parametrize("image_size", [256, 512])
def test_real_cuda_checkpoint_smoke(real_weights, image_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    root = Path(real_weights)
    checkpoint = root / {
        256: "256x256_diffusion_uncond.pt",
        512: "512x512_diffusion_uncond_finetune_008100.pt",
    }[image_size]
    secondary = root / "secondary_model_imagenet_2.pth"
    missing = [str(path) for path in (checkpoint, secondary) if not path.is_file()]
    if missing:
        pytest.skip("real checkpoint files are missing: " + ", ".join(missing))
    settings = normalise_settings(DEFAULTS, {
        "prompts": ["a blue glass sphere"], "weights": [1.0],
        "image_size": image_size, "width": image_size, "height": image_size,
        "steps": 2, "clip_models": ["ViTB32"], "cut_overview": 1,
        "cut_innercut": 0, "cutn_batches": 1, "cut_batch": 1,
        "device": "cuda", "precision": "bf16", "deterministic": True,
    })
    pixels, effective, _ = render(settings, checkpoint, secondary, progress=False)
    assert torch.isfinite(pixels).all()
    assert effective["actual_steps"] == 2


@pytest.mark.integration
def test_deterministic_mixed_clip_resize_backward_uses_cpu_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class Encoder(nn.Module):
        def encode_image(self, x):
            return x.mean((2, 3))

    bank = ClipBank.__new__(ClipBank)
    nn.Module.__init__(bank)
    bank.device = torch.device("cuda")
    bank.models = [Encoder().cuda()]
    bank.means = [torch.zeros(1, 3, 1, 1, device="cuda")]
    bank.stds = [torch.ones(1, 3, 1, 1, device="cuda")]
    bank.sizes = [6]
    bank.dtype = torch.float32
    prepare_reference_runtime(torch.device("cuda"))

    def gradient():
        source = torch.linspace(0, 1, 3 * 4 * 4).reshape(1, 3, 4, 4).requires_grad_(True)
        with render_rng(4, torch.device("cuda"), deterministic=True):
            bank.encode_cutouts(source, 0, deterministic=True).sum().backward()
        return source.grad

    assert torch.equal(gradient(), gradient())

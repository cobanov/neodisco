"""Vendored from openai/guided-diffusion (MIT), sampling path only.

Only the five modules needed to build and run the 2021 unconditional ImageNet models are
copied here: the UNet, the Gaussian diffusion, timestep respacing, and their two helper
modules. The rest of that repository pulls in mpi4py and blobfile for distributed training
and dataset streaming, which nothing here needs, and which make the package awkward to
install in 2026.

Upstream: https://github.com/openai/guided-diffusion
Licence: MIT, see LICENSE in this directory.
"""

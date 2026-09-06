# Studio typography

The studio deployment uses Foundry Gridnik Light (300), Regular (400), and Medium
(500), matching the existing vggtq and vgpu-umaps interfaces. Studio WOFF2 files
are provisioned privately into this directory and excluded from Git. The interface
falls back to the system monospace face when they are absent.

Expected names: `FoundryGridnik-Light.woff2`, `FoundryGridnik-Regular.woff2`,
`FoundryGridnik-Medium.woff2`.

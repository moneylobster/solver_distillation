"""DLMS: reimplementation of "Linear Multistep Solver Distillation for Fast
Sampling of Diffusion Models" (ICLR 2025).

This package reuses the diff-sampler toolbox (external/diff-sampler) for
pretrained-model loading, schedules and reference solvers; see bootstrap.py.
"""

from . import bootstrap  # noqa: F401  (sets up sys.path for amed-solver-main)

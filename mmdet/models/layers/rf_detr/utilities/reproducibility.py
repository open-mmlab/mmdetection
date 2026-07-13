# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Reproducibility helpers: seed all RNGs consistently."""

import random

import numpy as np
import torch


def seed_all(seed: int = 7) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch (CPU and all CUDA devices).  Also configures cuDNN to
    use deterministic algorithms and disables its auto-tuner so that results are reproducible across runs at the cost of
    a possible slight performance decrease.

    Under distributed data-parallel, ``random`` and NumPy are offset by the process rank so stochastic augmentations
    differ across ranks, while PyTorch is seeded identically so model initialization stays in sync.

    Args:
        seed: Integer seed value.  Defaults to ``7``.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

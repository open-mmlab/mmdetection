# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Drop-path / dropout schedule utilities."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def drop_scheduler(
    drop_rate: float,
    epochs: int,
    niter_per_ep: int,
    cutoff_epoch: int = 0,
    mode: Literal["standard", "early", "late"] = "standard",
    schedule: Literal["constant", "linear"] = "constant",
) -> NDArray[np.float64]:
    """Build a per-iteration drop-path or dropout rate schedule.

    ``"standard"`` mode: every iteration uses the same ``drop_rate``; ignored ``cutoff_epoch``, ``schedule``.
    ``"early"`` mode: drop rate applies during the first ``cutoff_epoch`` epochs; remaining epochs use zero drop rate.
        optional: schedule ``linear`` decay to zero over the first ``cutoff_epoch`` epochs.
    ``"late"`` mode: the first ``cutoff_epoch`` epochs use zero drop rate; remaining epochs use ``drop_rate``.

    Args:
        drop_rate: Target drop probability.
        epochs: Total number of training epochs.
        niter_per_ep: Number of optimizer steps per epoch.
        cutoff_epoch: Number of epochs in the initial schedule phase. Phases split at cutoff_epoch * niter_per_ep steps.
            Ignored when ``mode`` is ``"standard"``.
        mode: Scheduling strategy: ``"standard"``, ``"early"``, or ``"late"``.
        schedule: Shape of the initial schedule phase in ``"early"`` mode: ``"constant"`` or ``"linear"``.
            Ignored when ``mode`` is ``"standard"``; only ``"constant"`` is accepted for ``"late"`` mode.

    Returns:
        One-dimensional array of length ``epochs * niter_per_ep`` containing the drop rate per iteration.

    Raises:
        ValueError: If ``mode`` is not ``"standard"``, ``"early"``, or ``"late"``.
        ValueError: If ``epochs`` or ``niter_per_ep`` is less than ``1``.
        ValueError: If ``cutoff_epoch`` is not in the range ``[0, epochs]``.
        ValueError: If ``schedule`` is not ``"constant"`` or ``"linear"`` in ``"early"`` mode.
        NotImplementedError: If ``schedule`` is not ``"constant"`` in ``"late"`` mode.
    """
    if mode not in ("standard", "early", "late"):
        raise ValueError(f"mode must be 'standard', 'early', or 'late', got {mode!r}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if niter_per_ep < 1:
        raise ValueError(f"niter_per_ep must be >= 1, got {niter_per_ep}")
    total_iters = epochs * niter_per_ep
    if mode == "standard":
        return np.full(total_iters, drop_rate)
    if not 0 <= cutoff_epoch <= epochs:
        raise ValueError(f"cutoff_epoch must be in [0, {epochs}], but got {cutoff_epoch}")
    early_iters = cutoff_epoch * niter_per_ep
    result = np.zeros(total_iters, dtype=np.float64)

    if mode == "early":
        if schedule not in ("constant", "linear"):
            raise ValueError(f"schedule must be 'constant' or 'linear', got {schedule!r}")
        if schedule == "constant":
            result[:early_iters] = drop_rate
        else:
            result[:early_iters] = np.linspace(drop_rate, 0, early_iters)
    elif mode == "late":
        if schedule != "constant":
            raise NotImplementedError(
                f"schedule={schedule!r} is not supported for mode='late'; only 'constant' is implemented"
            )
        result[early_iters:] = drop_rate

    return result

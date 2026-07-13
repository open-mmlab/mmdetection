# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint conversion utilities for the PTL training stack.

Provides :func:`convert_legacy_checkpoint` to convert RF-DETR ``*.pth`` checkpoints (produced by the pre-PTL
``engine.py`` training loop) into the ``*.ckpt`` format expected by ``pytorch_lightning.Trainer``.

Auto-detection of legacy format at load time is handled by
:meth:`rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["convert_legacy_checkpoint"]


def convert_legacy_checkpoint(old_path: str, new_path: str) -> None:
    """Convert a legacy RF-DETR ``.pth`` checkpoint to PTL ``.ckpt`` format.

    Loads a checkpoint saved by the pre-PTL ``engine.py`` training loop and rewrites it in the structure expected by
    ``pytorch_lightning.Trainer``:

    * ``state_dict`` keys are prefixed with ``"model."`` to match the
      attribute path inside :class:`~rfdetr.training.module_model.RFDETRModelModule`.
    * ``args`` (``argparse.Namespace`` or ``dict``) is normalised to a plain
      ``dict`` and stored as ``hyper_parameters``.
    * ``legacy_checkpoint_format: True`` is written so
      :meth:`~rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint` can distinguish converted files from
      native PTL checkpoints.
    * If an ``ema_model`` key is present it is preserved verbatim under
      ``legacy_ema_state_dict`` for optional EMA weight restoration.

    Args:
        old_path: Path to the source legacy ``.pth`` checkpoint.
        new_path: Destination path for the converted ``.ckpt`` file.
    """
    # trust=True: this function converts internally-produced legacy .pth files;
    # allow pickle fallback if safe deserialization fails due to non-tensor/custom objects.
    from mmdet.models.layers.rf_detr.util.io import _safe_torch_load

    old: dict[str, Any] = _safe_torch_load(old_path, trust=True)

    if "model" not in old:
        raise ValueError(
            f"The checkpoint at {old_path!r} does not contain a 'model' key."
            " Only RF-DETR legacy .pth files produced by engine.py are supported."
        )

    args_obj = old.get("args")
    if isinstance(args_obj, dict):
        hyper_parameters: dict[str, Any] = args_obj
    elif args_obj is None:
        hyper_parameters = {}
    else:
        try:
            hyper_parameters = vars(args_obj)
        except TypeError:
            logger.warning(
                "Cannot extract hyper_parameters from args of type %s; storing empty dict.",
                type(args_obj).__name__,
            )
            hyper_parameters = {}

    new: dict[str, Any] = {
        "state_dict": {"model." + k: v for k, v in old["model"].items()},
        "epoch": old.get("epoch", 0),
        "global_step": 0,
        "hyper_parameters": hyper_parameters,
        "legacy_checkpoint_format": True,
    }

    if "ema_model" in old:
        # Preserve EMA weights under a dedicated key.  Callback-specific state
        # keys are framework-internal and must not be written here.
        new["legacy_ema_state_dict"] = old["ema_model"]

    torch.save(new, new_path)

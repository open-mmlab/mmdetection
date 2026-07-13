# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Safe checkpoint loading helpers.

This module provides :func:`_safe_torch_load`, a defense-in-depth wrapper around :func:`torch.load` that prevents
pickle-based remote code execution (CWE-502) when loading checkpoints from external or user-supplied sources.
"""

from __future__ import annotations

import argparse
import pickle
import types
import warnings
from pathlib import Path
from typing import Any

import torch

# No public API: _safe_torch_load is an underscore-private, security-sensitive
# helper imported directly by trusted internal callers, not for ad-hoc external use.
__all__: list[str] = []


def _safe_torch_load(path: str | Path, *, trust: bool = False) -> Any:
    """Load a PyTorch checkpoint as safely as possible.

    Tries progressively less restrictive deserialization strategies:

    1. ``weights_only=True`` (strict — only tensors and a small set of
       built-in scalars).
    2. Same as 1, but with ``argparse.Namespace`` and
       ``types.SimpleNamespace`` temporarily allowed via the
       :func:`torch.serialization.safe_globals` context manager so that legacy
       RF-DETR checkpoints that embed an ``args`` namespace can be loaded
       without falling back to pickle.  The allow-list is scoped to this call
       and does **not** permanently mutate global deserialization state.
    3. ``weights_only=False`` (full pickle) — allowed **only** when
       ``trust=True``, with a loud :class:`UserWarning`.  Never used for
       checkpoints received from external sources.

    Args:
        path: Path to the checkpoint file.
        trust: When ``True``, allow pickle deserialization as a last-resort
            fallback and emit a :class:`UserWarning`.  Set this only for
            checkpoint files produced by RF-DETR itself (e.g. during legacy
            checkpoint conversion) that may contain non-tensor Python
            objects that are not covered by the safe-globals list.

    Returns:
        The loaded checkpoint (usually a :class:`dict`).

    Raises:
        RuntimeError: When all safe loading strategies fail and
            ``trust=False``.  The error message suggests passing ``trust=True``
            (or ``trust_checkpoint=True`` at the ``RFDETR.from_checkpoint()``
            level) so the caller can make an informed decision.

    Examples:
        >>> import torch, tempfile, os
        >>> with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fh:
        ...     path = fh.name
        >>> torch.save({"model": {"weight": torch.tensor([1.0])}}, path)
        >>> ckpt = _safe_torch_load(path)
        >>> list(ckpt.keys())
        ['model']
        >>> os.unlink(path)
    """
    # ── Attempt 1: strict safe load ──────────────────────────────────────
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        pass

    # ── Attempt 2: scoped safe globals (no permanent process-global mutation) ─
    # argparse.Namespace and types.SimpleNamespace appear in RF-DETR checkpoints
    # saved by the pre-PTL engine.py training loop.  Allowing them is semantically
    # safe because they contain only primitive values.  Using the
    # ``safe_globals`` context manager scopes the allow-list to this single load
    # (instead of the process-global ``add_safe_globals``) and folds the retry
    # into one ``torch.load`` call rather than a separate double-load.
    try:
        with torch.serialization.safe_globals([argparse.Namespace, types.SimpleNamespace]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        pass

    # ── Attempt 3 (opt-in): full pickle ───────────────────────────────────
    if trust:
        warnings.warn(
            f"Loading checkpoint {str(path)!r} with weights_only=False. "
            "This allows arbitrary Python objects to be deserialized from the "
            "checkpoint file, which can execute malicious code if the file "
            "comes from an untrusted source. "
            "Only use trust=True for checkpoint files produced by RF-DETR itself.",
            UserWarning,
            stacklevel=3,
        )
        return torch.load(path, map_location="cpu", weights_only=False)

    raise RuntimeError(
        f"Failed to safely load checkpoint {str(path)!r}. "
        "The file likely contains custom Python objects that cannot be "
        "deserialized with weights_only=True. "
        "If you fully trust this checkpoint source, pass trust=True to "
        "_safe_torch_load() or trust_checkpoint=True to RFDETR.from_checkpoint()."
    )

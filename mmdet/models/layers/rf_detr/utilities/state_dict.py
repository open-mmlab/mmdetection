# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint and state-dict helpers."""

import os
import tempfile
from collections import OrderedDict
from typing import Any

from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

# PTL-compatible keys written by BestModelCallback; preserved by strip_checkpoint so
# checkpoint_best_total.pth can be used directly with trainer.fit(ckpt_path=...).
_PTL_COMPAT_KEYS = (
    "state_dict",
    "global_step",
    "pytorch-lightning_version",
    "loops",
    "optimizer_states",
    "lr_schedulers",
)


def _raise_patch_size_mismatch(ckpt_patch_size: int, model_patch_size: int) -> None:
    """Raise a descriptive ValueError for a patch_size incompatibility.

    Args:
        ckpt_patch_size: patch_size recorded in (or inferred from) the checkpoint.
        model_patch_size: patch_size the current model is configured with.

    Raises:
        ValueError: Always — describes the mismatch and how to resolve it.
    """
    raise ValueError(
        f"The checkpoint was trained with patch_size={ckpt_patch_size}, but the current model uses "
        f"patch_size={model_patch_size}. The checkpoint is incompatible with this model architecture. "
        "To resolve this, either instantiate/configure the model with the checkpoint's patch_size or "
        "use a checkpoint that was trained with the same patch_size as the current model."
    )


def _ckpt_args_get(args: Any, field: str, default: Any = None) -> Any:
    """Get a field from checkpoint ``"args"``, handling both dict and attribute access.

    New checkpoints (PTL training stack) store ``"args"`` as a plain ``dict`` (via ``TrainConfig.model_dump()``).
    Legacy checkpoints (pre-PTL engine or the pre-release PTL code) stored it as a ``Namespace``-like object.  This
    helper abstracts both so callers do not need to branch on the type.

    Args:
        args: The ``checkpoint["args"]`` value.
        field: Field name to retrieve.
        default: Value returned when the field is absent.

    Returns:
        The field value, or ``default`` if not found.
    """
    if isinstance(args, dict):
        return args.get(field, default)
    return getattr(args, field, default)


def _make_fit_loop_state(epoch: int) -> dict:
    """Build a minimal ``fit_loop`` state dict that restores the epoch counter.

    ``BestModelCallback`` stores ``trainer.current_epoch`` as ``"epoch"`` in the checkpoint.  That value is captured
    during ``on_validation_end``, which fires *before* the loop's epoch-end hooks increment the counter.  To resume
    training *after* that epoch, PTL's epoch-progress counter must be set to ``epoch + 1`` so that
    ``trainer.current_epoch == epoch + 1`` when the new ``trainer.fit()`` call begins.

    Optimizer and scheduler states are intentionally omitted — loading a ``.pth`` file starts a fresh optimizer for the
    new training phase.

    Args:
        epoch: The ``"epoch"`` value from the checkpoint (``trainer.current_epoch``
            at the time of ``on_validation_end``).

    Returns:
        A ``fit_loop`` state dict compatible with :meth:`pytorch_lightning.loops._FitLoop.load_state_dict`.
    """
    n = epoch + 1  # number of epochs fully completed after epoch `epoch` finishes
    zero4 = {"ready": 0, "started": 0, "processed": 0, "completed": 0}
    zero3 = {"ready": 0, "started": 0, "completed": 0}
    zero2 = {"ready": 0, "completed": 0}
    n4 = {"ready": n, "started": n, "processed": n, "completed": n}
    return {
        "state_dict": {},
        "epoch_loop.state_dict": {"_batches_that_stepped": 0},
        "epoch_loop.batch_progress": {
            "total": {**zero4},
            "current": {**zero4},
            "is_last_batch": False,
        },
        "epoch_loop.scheduler_progress": {
            "total": {**zero2},
            "current": {**zero2},
        },
        "epoch_loop.automatic_optimization.state_dict": {},
        "epoch_loop.automatic_optimization.optim_progress": {
            "optimizer": {
                "step": {
                    "total": {**zero2},
                    "current": {**zero2},
                },
                "zero_grad": {
                    "total": {**zero3},
                    "current": {**zero3},
                },
            }
        },
        "epoch_loop.manual_optimization.state_dict": {},
        "epoch_loop.manual_optimization.optim_step_progress": {
            "total": {**zero2},
            "current": {**zero2},
        },
        "epoch_loop.val_loop.state_dict": {},
        "epoch_loop.val_loop.batch_progress": {
            "total": {**zero4},
            "current": {**zero4},
            "is_last_batch": False,
        },
        "epoch_progress": {
            "total": {**n4},
            "current": {**n4},
        },
    }


def strip_checkpoint(checkpoint: str | os.PathLike[str]) -> None:
    """Strip a checkpoint file down to ``model``, ``args``, and PTL-compatible keys.

    Preserves ``model_name`` (when present) so that ``RFDETR.from_checkpoint()`` can still resolve the model class from
    the stripped file.  Also preserves ``rfdetr_version`` (when present) for provenance tracking.

    Also preserves ``state_dict``, ``global_step``, ``pytorch-lightning_version``, ``loops``, ``optimizer_states``, and
    ``lr_schedulers`` when present so the stripped checkpoint can still be used directly with
    ``trainer.fit(ckpt_path=...)``.

    Overwrites the file atomically so a partial write cannot corrupt it.

    Args:
        checkpoint: Path to the ``.pth`` checkpoint file to strip in place.
    """
    import torch

    from mmdet.models.layers.rf_detr.util.io import _safe_torch_load

    # `checkpoint_best_total.pth` is produced by local RF-DETR training and can
    # contain non-tensor metadata under "args" (e.g. `types.SimpleNamespace`).
    # PyTorch 2.6 changed `torch.load` default `weights_only=True`, which rejects
    # these objects. This utility intentionally operates on trusted, locally
    # produced checkpoints, so trust=True permits the full-pickle fallback.
    state_dict = _safe_torch_load(checkpoint, trust=True)
    new_state_dict = {
        "model": state_dict["model"],
        "args": state_dict["args"],
    }
    # Preserve model_name when present (#887).
    if "model_name" in state_dict:
        new_state_dict["model_name"] = state_dict["model_name"]
    # Preserve rfdetr_version when present for provenance tracking.
    if "rfdetr_version" in state_dict:
        new_state_dict["rfdetr_version"] = state_dict["rfdetr_version"]
    # Preserve PTL-compatible keys when present (written by BestModelCallback).
    for key in _PTL_COMPAT_KEYS:
        if key in state_dict:
            new_state_dict[key] = state_dict[key]
    # Backfill validate_loop/test_loop stubs so old checkpoints (saved before the stubs
    # were added to _build_checkpoint_payload) can be passed as ckpt_path to
    # trainer.validate() / trainer.test() without a KeyError from PTL restore_loops().
    if "loops" in new_state_dict:
        loops = new_state_dict["loops"]
        for loop_key in ("validate_loop", "test_loop"):
            if loop_key not in loops:
                loops[loop_key] = {"state_dict": {}}
    # Create the temp file in the destination directory so os.replace stays on the same filesystem (atomic).
    checkpoint_dir = os.path.dirname(os.path.abspath(os.fspath(checkpoint)))
    with tempfile.NamedTemporaryFile(dir=checkpoint_dir, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(new_state_dict, tmp_path)
        # Atomic replace avoids leaving a partially written checkpoint on save failures/interruption.
        os.replace(tmp_path, checkpoint)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def clean_state_dict(state_dict: dict[str, Any]) -> OrderedDict[str, Any]:
    """Remove the ``module.`` prefix added by ``DataParallel`` / ``DistributedDataParallel``.

    Args:
        state_dict: State dict potentially containing ``module.``-prefixed keys.

    Returns:
        New ``OrderedDict`` with ``module.`` stripped from all keys.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def remap_projector_to_cross_attn(state_dict: dict[str, Any], model: object) -> dict[str, Any]:
    """Clone backbone projector weights into ``cross_attn_projector`` for dual-projector models.

    Older checkpoints can contain only ``backbone.0.projector.*`` keys. Dual-projector models expect an additional
    ``backbone.0.cross_attn_projector.*`` branch; this helper seeds it by cloning projector weights when missing.

    Args:
        state_dict: Checkpoint model-state dictionary (mutated in place).
        model: Instantiated model used to detect whether dual projector mode is active.

    Returns:
        The same ``state_dict`` object for convenience.
    """
    backbone = model.backbone[0] if hasattr(model, "backbone") else None
    if backbone is None:
        return state_dict

    # Real dual-projector backbones expose cross_attn_projector; compatibility shims may only carry dual_projector.
    has_cross_attn_projector = getattr(backbone, "cross_attn_projector", None) is not None
    dual_projector_enabled = bool(getattr(backbone, "dual_projector", False)) or has_cross_attn_projector
    if not dual_projector_enabled:
        return state_dict

    if any(key.startswith("backbone.0.cross_attn_projector.") for key in state_dict):
        return state_dict

    projector_keys = {key: value for key, value in state_dict.items() if key.startswith("backbone.0.projector.")}
    if not projector_keys:
        return state_dict

    logger.info(
        "Cloning %d backbone projector key(s) into cross_attn_projector for dual-projector compatibility.",
        len(projector_keys),
    )
    for key, value in projector_keys.items():
        new_key = key.replace("backbone.0.projector.", "backbone.0.cross_attn_projector.", 1)
        state_dict[new_key] = value.clone()

    return state_dict


def validate_checkpoint_compatibility(checkpoint: dict[str, Any], model_args: Any) -> None:
    """Validate that a checkpoint is compatible with the model configuration.

    Checks for mismatches in ``segmentation_head`` and ``patch_size`` between the checkpoint's saved training arguments
    and the current model configuration. Raises a descriptive :class:`ValueError` before ``load_state_dict`` fires so
    that users receive a clear, actionable message instead of a cryptic tensor size mismatch error.

    If either side is missing an attribute (e.g. a legacy checkpoint saved before
        ``segmentation_head`` or ``patch_size`` was added to ``args``), that specific
        check is skipped silently — this preserves backwards compatibility with pre-existing checkpoints.

    Args:
        checkpoint: Loaded checkpoint dictionary, expected to contain an optional
            ``"args"`` key with training namespace attributes or a plain dict.
        model_args: Namespace (e.g. ``types.SimpleNamespace``) with at least
            ``segmentation_head`` and ``patch_size`` attributes describing the current model.

    Raises:
        ValueError: If ``segmentation_head`` or ``patch_size`` in the checkpoint
            args do not match those of the model, or if the ``patch_size`` inferred from the DINOv2 projection weight
            shape differs from ``model_args.patch_size`` when no explicit ``args.patch_size`` is present.

    Note:
        This helper does not mutate ``model_args``. It emits ``logger.warning`` (not an exception) for class-count
        mismatches so that callers can still proceed with reinitialization or weight loading.

        When ``"args"`` is absent or ``args.patch_size`` is not set, a fallback infers ``patch_size`` from the DINOv2
        patch-embedding projection weight shape (key
        ``backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight``). This fallback **can raise**
        :class:`ValueError` on a mismatch, providing a clear error before the cryptic :class:`RuntimeError` from
        :meth:`~torch.nn.Module.load_state_dict` would otherwise fire. For all other attributes (e.g.
        ``segmentation_head``), if either side is missing, that check is skipped silently — preserving backward
        compatibility.

        Two class-count scenarios are distinguished:

        * Backbone pretrain: the checkpoint head was trained with more classes
          than the current ``model_args.num_classes``. In this case the detection head is typically reinitialized or
          trimmed externally to match the configured number of classes.
        * Fine-tuned checkpoint: the checkpoint head was trained with fewer
          classes than the current ``model_args.num_classes``. If you intend to reuse the checkpoint's classification
          head as-is, set ``model_args.num_classes`` to ``ckpt_num_classes - 1`` (the value reported in the warning)
          before loading the state dict to align the configuration and silence the warning.
    """
    # Emit actionable class-count mismatch warning early, before any reinit happens.
    ckpt_class_bias = checkpoint.get("model", {}).get("class_embed.bias", None)
    if ckpt_class_bias is not None:
        ckpt_num_classes = ckpt_class_bias.shape[0]
        model_num_classes: int | None = getattr(model_args, "num_classes", None)
        if model_num_classes is not None and ckpt_num_classes != model_num_classes + 1:
            if model_num_classes + 1 < ckpt_num_classes:
                # Backbone pretrain scenario: checkpoint has more classes, head will be trimmed.
                logger.warning(
                    "Checkpoint has %d classes but model is configured for %d. "
                    "The detection head will be re-initialized to %d classes.",
                    ckpt_num_classes - 1,
                    model_num_classes,
                    model_num_classes,
                )
            else:
                # Fine-tuned checkpoint loaded with wrong (larger) num_classes.
                logger.warning(
                    "Checkpoint has %d classes but model is configured for %d. "
                    "Using checkpoint class count (%d). "
                    "Pass num_classes=%d to suppress this warning.",
                    ckpt_num_classes - 1,
                    model_num_classes,
                    ckpt_num_classes - 1,
                    ckpt_num_classes - 1,
                )

    # Infer patch_size from the patch-embedding projection weight only as a fallback
    # when the checkpoint has no explicit args.patch_size (e.g., COCO pretrained
    # release weights that only store "model").
    # Conv2d projection shape is [out_channels, in_channels, kernel_h, kernel_w];
    # kernel_h == patch_size for square kernels. Raises before load_state_dict fires,
    # replacing the otherwise-cryptic "size mismatch" RuntimeError. Regression: #965.
    # NOTE: key path is DINOv2-specific; non-DINOv2 backbones simply won't have this key
    # and the check is silently skipped, preserving backward compatibility.
    _ckpt_args = checkpoint.get("args")
    _ckpt_patch_size_from_args: int | None = None
    if _ckpt_args is not None:
        _ckpt_patch_size_from_args = _ckpt_args_get(_ckpt_args, "patch_size")

    if _ckpt_patch_size_from_args is None:
        _patch_proj_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        _ckpt_proj_w = checkpoint.get("model", {}).get(_patch_proj_key)
        _ckpt_proj_shape = getattr(_ckpt_proj_w, "shape", None)
        if _ckpt_proj_shape is not None and len(_ckpt_proj_shape) == 4 and _ckpt_proj_shape[2] == _ckpt_proj_shape[3]:
            _inferred_ps = int(_ckpt_proj_shape[-1])
            _model_ps: int | None = getattr(model_args, "patch_size", None)
            if _model_ps is not None and _inferred_ps != _model_ps:
                _raise_patch_size_mismatch(_inferred_ps, _model_ps)
    if "args" not in checkpoint:
        return

    ckpt_args = checkpoint["args"]
    ckpt_segmentation_head: bool | None = _ckpt_args_get(ckpt_args, "segmentation_head")
    model_segmentation_head: bool | None = getattr(model_args, "segmentation_head", None)

    if (
        ckpt_segmentation_head is not None
        and model_segmentation_head is not None
        and ckpt_segmentation_head != model_segmentation_head
    ):
        if ckpt_segmentation_head:
            raise ValueError(
                "The checkpoint was trained with a segmentation head, but the current model does not have one. "
                "Load the weights into a segmentation model (e.g. RFDETRSegNano) instead of a detection model."
            )
        else:
            raise ValueError(
                "The current model has a segmentation head, but the checkpoint was trained without one. "
                "Load the weights into a detection model (e.g. RFDETRNano) instead of a segmentation model."
            )

    ckpt_patch_size: int | None = _ckpt_args_get(ckpt_args, "patch_size")
    model_patch_size: int | None = getattr(model_args, "patch_size", None)
    if ckpt_patch_size is not None and model_patch_size is not None and ckpt_patch_size != model_patch_size:
        _raise_patch_size_mismatch(ckpt_patch_size, model_patch_size)

    ckpt_keypoint_head: bool | None = _ckpt_args_get(ckpt_args, "use_grouppose_keypoints")
    model_keypoint_head: bool | None = getattr(model_args, "use_grouppose_keypoints", None)

    if ckpt_keypoint_head is not None and model_keypoint_head is not None and ckpt_keypoint_head != model_keypoint_head:
        if ckpt_keypoint_head:
            raise ValueError(
                "The checkpoint was trained with a keypoint head, but the current model does not have one. "
                "Load the weights into a keypoint model (e.g. RFDETRKeypointPreview) instead of a detection model."
            )
        else:
            raise ValueError(
                "The current model has a keypoint head, but the checkpoint was trained without one. "
                "Load the weights into a detection model (e.g. RFDETRNano) instead of a keypoint model."
            )

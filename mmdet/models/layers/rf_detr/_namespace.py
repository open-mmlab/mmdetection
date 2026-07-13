# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Package-private helper: build a self-contained namespace from Pydantic configs.

Replaces the previous shim in ``_args.py`` that called the deprecated ``populate_args()`` function from ``main.py``.
This module has zero dependency on ``main.py`` and can survive its deletion.
"""

import dataclasses
import types

from deprecate import deprecated

from mmdet.models.layers.rf_detr.config import ModelConfig, TrainConfig
from mmdet.models.layers.rf_detr.models._defaults import MODEL_DEFAULTS, ModelDefaults

# Fields forwarded from ModelConfig into the namespace.
# Excludes cls_loss_coef (handled by transitional override logic below).
_MC_NAMESPACE_FIELDS = {
    "amp",
    "backbone_lora",
    "bbox_reparam",
    "ca_nheads",
    "dec_layers",
    "dec_n_points",
    "device",
    "encoder",
    "freeze_encoder",
    "gradient_checkpointing",
    "group_detr",
    "hidden_dim",
    "ia_bce_loss",
    "layer_norm",
    "lite_refpoint_refine",
    "mask_downsample_ratio",
    "num_channels",
    "num_classes",
    "num_queries",
    "num_select",
    "num_windows",
    "out_feature_indexes",
    "patch_size",
    "positional_encoding_size",
    "pretrain_weights",
    "projector_scale",
    "resolution",
    "sa_nheads",
    "segmentation_head",
    "use_grouppose_keypoints",
    "keypoint_cross_attn",
    "inter_instance_kp_attn",
    "grouppose_keypoint_dim_downscale",
    "dual_projector",
    "dual_projector_kp_only",
    "num_keypoints_per_class",
    "num_decoder_registers",
    "postprocess_trace_alpha",
    "two_stage",
}

# TrainConfig fields NOT forwarded to the legacy namespace.
# _TC_NAMESPACE_FIELDS is derived as: all TrainConfig fields minus this set.
#
# Excluded categories:
#   - Explicit transformations: handled with custom logic in _namespace_from_configs.
#   - Deprecated TC architecture copies: ModelConfig wins (see _MC_NAMESPACE_FIELDS).
#   - PTL Trainer / DDP, logger flags, auto-batch probe, DataModule knobs:
#     not consumed by legacy builders.
_TC_NON_NAMESPACE_FIELDS = {
    # Explicit transformations.
    "resume",
    "seed",
    "cls_loss_coef",
    # Deprecated TC architecture copies — ModelConfig wins.
    "group_detr",
    "ia_bce_loss",
    "segmentation_head",
    "num_select",
    # PTL Trainer / DDP.
    "accelerator",
    "strategy",
    "devices",
    "num_nodes",
    # Logger flags.
    "tensorboard",
    "wandb",
    "mlflow",
    "clearml",
    "project",
    "run",
    # Auto-batch probe.
    "auto_batch_target_effective",
    "auto_batch_max_targets_per_image",
    "auto_batch_ema_headroom",
    # PTL-only Trainer / DataModule / LR-scheduler knobs.
    "progress_bar",
    "compute_train_metrics",
    "run_test",
    "dont_save_weights",
    "pin_memory",
    "persistent_workers",
    "lr_scheduler",
    "lr_min_factor",
    # Dataset class labels.
    "class_names",
}

# Derived: all TrainConfig fields not in _TC_NON_NAMESPACE_FIELDS.
_TC_NAMESPACE_FIELDS = set(TrainConfig.model_fields) - _TC_NON_NAMESPACE_FIELDS


def _namespace_from_configs(
    model_config: ModelConfig,
    train_config: TrainConfig,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> types.SimpleNamespace:
    """Build a ``types.SimpleNamespace`` from configs and architectural defaults.

    This is the internal implementation behind :func:`build_namespace`. Extracting it allows config-native builder
    functions to construct a namespace without going through the public ``build_namespace()`` API while still accepting
    overridable defaults.

    This function is used by multiple modules as the transitional namespace
    bridge: :func:`rfdetr.models.build_model_from_config`, :func:`rfdetr.models.build_criterion_from_config`, and
    :func:`rfdetr.detr._build_model_context` all call it directly to avoid the public ``build_namespace()`` shim.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants.  Defaults to :data:`MODEL_DEFAULTS`.

    Returns:
        ``types.SimpleNamespace`` compatible with ``build_model``, ``build_criterion_and_postprocessors``, and
        ``build_dataset``.
    """
    mc = model_config
    tc = train_config
    d = defaults
    train_fields_set = getattr(tc, "model_fields_set", set())
    model_fields_set = getattr(mc, "model_fields_set", set())
    # Transitional compatibility: during deprecation, preserve explicit
    # ModelConfig.cls_loss_coef values when TrainConfig does not set one.
    cls_loss_coef = (
        tc.cls_loss_coef
        if "cls_loss_coef" in train_fields_set or "cls_loss_coef" not in model_fields_set
        else mc.cls_loss_coef
    )

    return types.SimpleNamespace(
        **{
            # Architectural defaults — 35 constants not exposed in ModelConfig/TrainConfig.
            **dataclasses.asdict(d),
            # TrainConfig: fields consumed by legacy builders (PTL, logger, auto-batch
            # fields excluded; see _TC_NAMESPACE_FIELDS).  Architecture copies
            # (group_detr, num_select, …) are intentionally absent — mc wins below.
            **tc.model_dump(include=set(_TC_NAMESPACE_FIELDS)),
            # ModelConfig: wins over tc for overlapping architecture params
            # (group_detr, ia_bce_loss, segmentation_head, num_select).
            **mc.model_dump(include=set(_MC_NAMESPACE_FIELDS)),
            # Segmentation extras (SegmentationTrainConfig only — absent from base TrainConfig).
            "mask_ce_loss_coef": getattr(tc, "mask_ce_loss_coef", 5.0),
            "mask_dice_loss_coef": getattr(tc, "mask_dice_loss_coef", 5.0),
            "mask_point_sample_ratio": getattr(tc, "mask_point_sample_ratio", 16),
            # Transformations: fields requiring a default sentinel or transitional priority.
            "cls_loss_coef": cls_loss_coef,
            "resume": tc.resume or "",
            "seed": tc.seed if tc.seed is not None else 42,
        }
    )


@deprecated(target=_namespace_from_configs, deprecated_in="1.7.0", remove_in="1.9.0")
def build_namespace(model_config: ModelConfig, train_config: TrainConfig) -> types.SimpleNamespace:
    """Build a ``types.SimpleNamespace`` from Pydantic model and train configs.

    .. deprecated:: 1.7.0
        ``build_namespace`` is a backward-compatibility shim with no remaining internal callers.
        Deprecated since v1.7.0, will be removed in v1.9.0. Use the config-native builders instead:

        - :func:`rfdetr.models.build_model_from_config` — replaces
          ``build_model(build_namespace(mc, tc))``
        - :func:`rfdetr.models.build_criterion_from_config` — replaces
          ``build_criterion_and_postprocessors(build_namespace(mc, tc))``
        - :func:`rfdetr._namespace._namespace_from_configs` — for the rare
          case where a raw namespace is still required (e.g. ``build_dataset``)

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.

    Returns:
        ``types.SimpleNamespace`` compatible with ``build_model``, ``build_criterion_and_postprocessors``, and
        ``build_dataset``.
    """
    ...

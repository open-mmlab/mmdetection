# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared type protocols for RF-DETR model builder functions.

The ``BuilderArgs`` protocol documents the minimum attribute set consumed by ``build_model()``, ``build_backbone()``,
``build_transformer()``, and ``build_criterion_and_postprocessors()``.  It is satisfied structurally by any object that
exposes the required attributes — including the ``SimpleNamespace`` produced by
:func:`rfdetr._namespace.build_namespace` and, after Item #1 is complete, by ``ModelConfig``/``TrainConfig`` directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BuilderArgs(Protocol):
    """Protocol satisfied by both ``ModelConfig``-based Namespaces and raw ``SimpleNamespace`` objects.

    This documents the minimum attribute set consumed by ``build_model()``, ``build_backbone()``,
    ``build_transformer()``, and ``build_criterion_and_postprocessors()``.  All attributes that appear in
    ``build_namespace()`` hardcoded defaults are included.

    Note:
        Python 3.10/3.11 runtime ``isinstance()`` checks with ``@runtime_checkable`` only verify callable (method)
        presence, not data-attribute presence.  Full structural enforcement requires Python
        3.12+ or a static type checker (mypy / pyright).
    """

    # --- Architecture ---
    encoder: str
    out_feature_indexes: list[int]
    drop_path: float
    dec_layers: int
    freeze_encoder: bool
    backbone_lora: bool
    two_stage: bool
    use_grouppose_keypoints: bool
    keypoint_cross_attn: bool
    inter_instance_kp_attn: bool
    grouppose_keypoint_dim_downscale: int
    dual_projector: bool
    dual_projector_kp_only: bool
    num_keypoints_per_class: list[int]
    num_decoder_registers: int
    projector_scale: list[str]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool
    lite_refpoint_refine: bool
    layer_norm: bool
    amp: bool
    num_classes: int
    pretrain_weights: str | None
    device: str
    resolution: int
    group_detr: int
    gradient_checkpointing: bool
    positional_encoding_size: int
    ia_bce_loss: bool
    cls_loss_coef: float
    segmentation_head: bool
    mask_downsample_ratio: int
    num_queries: int
    num_select: int
    # --- Legacy / hardcoded defaults (present on Namespace, absent on raw configs) ---
    vit_encoder_num_layers: int
    window_block_indexes: list[int] | None
    position_embedding: str
    rms_norm: bool
    force_no_pretrain: bool
    dim_feedforward: int
    use_cls_token: bool
    pretrained_encoder: str | None
    backbone_only: bool
    encoder_only: bool
    # --- Criterion ---
    # Note: `decoder_norm`, `dropout`, and `num_feature_levels` are consumed by
    # `build_transformer()` (called inside `build_model()`) but are intentionally
    # absent here.  They are hardcoded constants computed/assigned inside
    # `build_namespace()` (e.g. `num_feature_levels = len(projector_scale)`) and
    # are never read from external callers — exposing them in the Protocol would
    # mislead consumers into thinking they must be supplied.
    aux_loss: bool
    focal_alpha: float
    bbox_loss_coef: float
    giou_loss_coef: float
    set_cost_class: float
    set_cost_bbox: float
    set_cost_giou: float
    use_varifocal_loss: bool
    use_position_supervised_loss: bool
    sum_group_losses: bool
    keypoint_l1_loss_coef: float
    keypoint_findable_loss_coef: float
    keypoint_visible_loss_coef: float
    keypoint_nll_loss_coef: float
    mask_ce_loss_coef: float
    mask_dice_loss_coef: float
    mask_point_sample_ratio: int

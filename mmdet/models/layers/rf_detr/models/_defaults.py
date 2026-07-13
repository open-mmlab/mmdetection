# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Hardcoded architectural constants not exposed in ModelConfig or TrainConfig.

These values correspond to the ``build_namespace()`` defaults in ``_namespace.py`` that have no corresponding config
field.  Making them explicit in a frozen dataclass enables testing, documentation, and (future) overrides without
touching config validation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    """Hardcoded architectural constants not exposed in ModelConfig or TrainConfig.

    These values mirror the legacy ``build_namespace()`` hardcoded section implemented in ``_namespace.py``. Making them
    explicit enables testing and future overrides without touching config validation.

    Note:
        ``ModelDefaults`` is public API as of v1.7.  Fields that represent true architectural decisions (e.g.
        ``dim_feedforward``, ``aux_loss``) will be promoted to ``ModelConfig`` or ``TrainConfig`` in future phases;
        field names and defaults may change across minor versions during this transitional period.

    Attributes:
        drop_mode: Drop-path mode used during training.
        drop_schedule: Schedule type for drop-path rate.
        cutoff_epoch: Epoch at which drop-path schedule resets.
        pretrained_encoder: Path/URL to a pretrained encoder checkpoint.
        pretrain_exclude_keys: Keys to exclude when loading pretrained weights.
        pretrain_keys_modify_to_load: Key remapping rules for pretrained weights.
        pretrained_distiller: Path/URL to a distillation teacher checkpoint.
        vit_encoder_num_layers: Number of layers in the ViT encoder.
        window_block_indexes: Indices of encoder layers using window attention.
        position_embedding: Type of positional embedding (``'sine'``).
        rms_norm: Whether to use RMSNorm instead of LayerNorm.
        force_no_pretrain: Force-disable pretrain weight loading.
        dim_feedforward: FFN hidden dimension in decoder layers.
        decoder_norm: Normalization type in decoder (``'LN'``).
        freeze_batch_norm: Whether to freeze batch-norm layers.
        use_cls_token: Whether to prepend a CLS token to the encoder.
        encoder_only: Build encoder only (no decoder).
        backbone_only: Build backbone only (no encoder or decoder).
        aux_loss: Whether to compute auxiliary losses at intermediate layers.
        focal_alpha: Alpha parameter for focal loss.
        set_cost_class: Classification cost weight for the matcher.
        set_cost_bbox: L1 bbox cost weight for the matcher.
        set_cost_giou: GIoU cost weight for the matcher.
        bbox_loss_coef: Bbox regression loss coefficient.
        giou_loss_coef: GIoU loss coefficient.
        sum_group_losses: Whether to sum (vs. average) group-DETR losses.
        use_varifocal_loss: Whether to use varifocal loss instead of focal loss.
        use_position_supervised_loss: Whether to use position-supervised loss.
        print_freq: Logging frequency (steps).
        do_benchmark: Whether to run in benchmark/profiling mode.
        dropout: Dropout rate in the decoder.
        coco_path: Path to a COCO dataset root (legacy).
        dont_save_weights: Disable checkpoint saving.
        start_epoch: Epoch to resume training from.
        eval: Whether to run in eval-only mode.
        world_size: Number of distributed processes.
        dist_url: URL for distributed initialisation.
        lr_scheduler: Learning-rate scheduler type.
        lr_min_factor: Minimum LR factor for cosine annealing.
        subcommand: CLI subcommand (legacy).
    """

    drop_mode: str = "standard"
    drop_schedule: str = "constant"
    cutoff_epoch: int = 0
    pretrained_encoder: str | None = None
    pretrain_exclude_keys: list[str] | None = None
    pretrain_keys_modify_to_load: dict[str, str] | None = None
    pretrained_distiller: str | None = None
    vit_encoder_num_layers: int = 12
    window_block_indexes: list[int] | None = None
    position_embedding: str = "sine"
    rms_norm: bool = False
    force_no_pretrain: bool = False
    dim_feedforward: int = 2048
    decoder_norm: str = "LN"
    freeze_batch_norm: bool = False
    use_cls_token: bool = False
    encoder_only: bool = False
    backbone_only: bool = False
    aux_loss: bool = True
    focal_alpha: float = 0.25
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    sum_group_losses: bool = False
    use_varifocal_loss: bool = False
    use_position_supervised_loss: bool = False
    print_freq: int = 10
    do_benchmark: bool = False
    dropout: float = 0.0
    coco_path: str | None = None
    dont_save_weights: bool = False
    start_epoch: int = 0
    eval: bool = False
    world_size: int = 1
    dist_url: str = "env://"
    lr_scheduler: str = "step"
    lr_min_factor: float = 0.0
    subcommand: str | None = None


MODEL_DEFAULTS = ModelDefaults()

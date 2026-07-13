# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Functions to get params dict."""

from typing import Any, cast

from torch import nn

from mmdet.models.layers.rf_detr.models.backbone import Joiner
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: parameter name.
        lr_decay_rate: base lr decay rate.
        num_layers: number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    # NOTE: near-duplicate of get_dinov2_lr_decay_rate in models/backbone/backbone.py (same formula,
    # different layer-key pattern: this matches ".blocks.", that matches ".layer.").
    # If updating this formula, update the sibling too.
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    logger.debug(f"name: {name}, lr_decay: {lr_decay_rate ** (num_layers + 1 - layer_id)}")
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Calculate weight decay rate for different ViT parameters.

    Args:
        name: parameter name.
        weight_decay_rate: base weight decay rate.

    Returns:
        weight decay rate for the given parameter.
    """
    if ("gamma" in name) or ("pos_embed" in name) or ("rel_pos" in name) or ("bias" in name) or ("norm" in name):
        weight_decay_rate = 0.0
    logger.debug(f"name: {name}, weight_decay rate: {weight_decay_rate}")
    return weight_decay_rate


def get_param_dict(args: Any, model_without_ddp: nn.Module) -> list[dict[str, Any]]:
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = cast("Any", model_without_ddp.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [p for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad]

    decoder_param_lr_pairs = [{"params": param, "lr": args.lr * args.lr_component_decay} for param in decoder_params]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters()
        if (n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [{"params": param, "lr": args.lr} for param in other_params]

    final_param_dicts = other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs

    return final_param_dicts

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Backbone modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mmdet.models.layers.rf_detr.models.backbone.base import BackboneBase
from mmdet.models.layers.rf_detr.models.backbone.dinov2 import DinoV2
from mmdet.models.layers.rf_detr.models.backbone.projector import MultiScaleProjector
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.tensors import NestedTensor

logger = get_logger()

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str | None = None,
        window_block_indexes: list[int] | None = None,
        drop_path: float = 0.0,
        out_channels: int = 256,
        out_feature_indexes: list[int] | None = None,
        projector_scale: list[str] | None = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 0,
        dual_projector: bool = False,
    ) -> None:
        super().__init__()
        # an example name here would be "dinov2_base" or "dinov2_registers_windowed_base"
        # if "registers" is in the name, then use_registers is set to True, otherwise it is set to False
        # similarly, if "windowed" is in the name, then use_windowed_attn is set to True, otherwise it is set to False
        # the last part of the name should be the size
        # and the start should be dinov2
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        # name_parts[-1]
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, (
            "name should be dinov2, then either registers, windowed, both, or none, then the size"
        )
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert self.projector_scale is not None and len(self.projector_scale) > 0
        # x[0]
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.cross_attn_projector = (
            MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
            )
            if dual_projector
            else None
        )

        self._export = False

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign,assignment]

        if not hasattr(self.encoder, "merge_and_unload"):
            return

        try:
            from peft import PeftModel
        except ModuleNotFoundError:
            logger.warning("peft is not installed; skipping LoRA weight merging during export.")
            return
        except ImportError as exc:
            logger.warning("Failed to import PeftModel from peft during export: %s", exc)
            raise

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights")
            self.encoder = self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], list[NestedTensor] | None]:
        """"""
        # (H, W, B, C)
        raw_feats = self.encoder(tensor_list.tensors)
        feats = self.projector(raw_feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))

        cross_attn_out = None
        if self.cross_attn_projector is not None:
            cross_attn_out = []
            cross_attn_feats = self.cross_attn_projector(raw_feats)
            for feat in cross_attn_feats:
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
                cross_attn_out.append(NestedTensor(feat, mask))

        return out, cross_attn_out

    def forward_export(self, tensors: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor] | None]:
        raw_feats = self.encoder(tensors)
        feats = self.projector(raw_feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)

        cross_attn_feats = None
        if self.cross_attn_projector is not None:
            cross_attn_feats = list(self.cross_attn_projector(raw_feats))

        return out_feats, out_masks, cross_attn_feats

    def get_named_param_lr_pairs(self, args: Any, prefix: str = "backbone.0") -> dict[str, dict[str, Any]]:
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Calculate lr decay rate for different ViT blocks.

    Args:
        name: Parameter name.
        lr_decay_rate: Base lr decay rate.
        num_layers: Number of ViT blocks.

    Returns:
        Lr decay rate for the given parameter.
    """
    # NOTE: near-duplicate of get_vit_lr_decay_rate in training/param_groups.py (same formula,
    # different layer-key pattern: this matches ".layer.", that matches ".blocks.").
    # If updating this formula, update the sibling too.
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate

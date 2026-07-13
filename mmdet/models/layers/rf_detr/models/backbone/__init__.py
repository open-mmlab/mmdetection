# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Any

from torch import Tensor, nn

from mmdet.models.layers.rf_detr.models.backbone.backbone import Backbone
from mmdet.models.layers.rf_detr.models.position_encoding import build_position_encoding
from mmdet.models.layers.rf_detr.utilities.tensors import NestedTensor


class Joiner(nn.Sequential):
    def __init__(self, backbone: Backbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], list[Tensor], list[NestedTensor] | None]:
        """"""
        result = self[0](tensor_list)
        if isinstance(result, tuple):
            x, cross_attn_x = result
        else:
            x, cross_attn_x = result, None
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos, cross_attn_x

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign,assignment]
        for name, m in self.named_modules():
            if hasattr(m, "export") and callable(m.export) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward_export(self, inputs: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor] | None]:
        result = self[0](inputs)
        if len(result) == 3:
            feats, masks, cross_attn_feats = result
        else:
            feats, masks = result
            cross_attn_feats = None
        poss = []
        for feat, mask in zip(feats, masks):
            pos = self[1](mask, align_dim_orders=False).to(feat.dtype)
            if cross_attn_feats is None and pos.ndim == 4 and pos.shape[1] == 1:
                pos = pos[:, 0]
            poss.append(pos)
        return feats, masks, poss, cross_attn_feats


def build_backbone(
    encoder: str,
    vit_encoder_num_layers: int,
    pretrained_encoder: str | None,
    window_block_indexes: list[Any] | None,
    drop_path: float,
    out_channels: int,
    out_feature_indexes: list[Any] | None,
    projector_scale: list[Any] | None,
    use_cls_token: bool,
    hidden_dim: int,
    position_embedding: str,
    freeze_encoder: bool,
    layer_norm: bool,
    target_shape: tuple[int, int],
    rms_norm: bool,
    backbone_lora: bool,
    force_no_pretrain: bool,
    gradient_checkpointing: bool,
    load_dinov2_weights: bool,
    patch_size: int,
    num_windows: int,
    positional_encoding_size: int,
    dual_projector: bool = False,
) -> Joiner:
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding_module = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        dual_projector=dual_projector,
    )

    model = Joiner(backbone, position_embedding_module)
    return model

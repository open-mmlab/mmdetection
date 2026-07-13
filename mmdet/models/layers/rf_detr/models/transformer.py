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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Transformer class."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from mmdet.models.layers.rf_detr.models._types import BuilderArgs
from mmdet.models.layers.rf_detr.models.heads.keypoints import ConditionalQueryInitializer
from mmdet.models.layers.rf_detr.models.math import MLP
from mmdet.models.layers.rf_detr.models.ops.modules import MSDeformAttn


def _safe_multinormalize(dim: int) -> int:
    """Clamp a MultiheadAttention head count to at least one."""
    return max(1, dim)


def gen_sineembed_for_position(pos_tensor: Tensor, dim: int = 128) -> Tensor:
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


def gen_encoder_output_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor | None = None,
    spatial_shapes: Sequence[tuple[int, int]] | Tensor | None = None,
    unsigmoid: bool = True,
) -> tuple[Tensor, Tensor]:
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    proposals = []
    _cur = 0
    batch_size, _, _ = memory.shape
    assert spatial_shapes is not None
    for lvl, (height, width) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            # reshape(-1, ...) infers batch dynamically in ONNX instead of baking it in as constants.
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + height * width)].reshape(batch_size, height, width, 1)

            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            # Avoid baking constants in ONNX.
            valid_height = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + height
            valid_width = torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + width

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # height, width, 2

        # Keep symbolic batch in ONNX.
        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).reshape(-1, 1, 1, 2)
        proposals_grid = (grid.unsqueeze(0) + 0.5) / scale.float()

        wh = torch.ones_like(proposals_grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((proposals_grid, wh), -1).reshape(batch_size, -1, 4)
        proposals.append(proposal)
        _cur += height * width

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):
    """Transformer with optional GroupPose keypoint decoder stream support."""

    def __init__(
        self,
        d_model: int = 512,
        sa_nhead: int = 8,
        ca_nhead: int = 8,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        lite_refpoint_refine: bool = False,
        decoder_norm_type: str = "LN",
        bbox_reparam: bool = False,
        use_grouppose_keypoints: bool = False,
        num_keypoints_per_class: list[int] | None = None,
        grouppose_keypoint_dim_downscale: int = 1,
        keypoint_cross_attn: bool = True,
        inter_instance_kp_attn: bool = False,
        num_registers: int = 0,
        dual_projector_kp_only: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = None
        self.enc_out_class_embed: nn.ModuleList | None = None
        self.enc_out_bbox_embed: nn.ModuleList | None = None
        self.enc_out_keypoint_embed: nn.ModuleList | None = None

        self.use_grouppose_keypoints = use_grouppose_keypoints
        self.dual_projector_kp_only = dual_projector_kp_only
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.num_registers = num_registers

        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
            enable_keypoint_processing=use_grouppose_keypoints,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
            keypoint_cross_attn=keypoint_cross_attn,
            inter_instance_kp_attn=inter_instance_kp_attn,
        )
        assert decoder_norm_type in ["LN", "Identity"]
        norm_ctors: dict[str, Callable[[int], nn.Module]] = {
            "LN": lambda channels: nn.LayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm_ctors[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
            enable_keypoint_processing=use_grouppose_keypoints,
            num_keypoints_per_class=self.num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=grouppose_keypoint_dim_downscale,
        )

        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(group_detr)])

            if use_grouppose_keypoints and self.num_keypoints_per_class:
                total_keypoints = sum(self.num_keypoints_per_class)
                if total_keypoints > 0:
                    keypoint_dim = d_model // grouppose_keypoint_dim_downscale
                    self.keypoint_query_initializer = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    self.keypoint_query_initializer_enc = ConditionalQueryInitializer(
                        d_model, total_keypoints, out_dim=keypoint_dim
                    )
                    self.enc_out_keypoint_embed = nn.ModuleList(
                        [MLP(keypoint_dim, d_model, keypoint_dim, 2) for _ in range(group_detr)]
                    )

        self._reset_parameters()

        # Register tokens used by GroupPose path.
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.empty(num_registers, d_model).normal_())
            self.register_ref_points = nn.Parameter(torch.zeros(num_registers, 4))

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False

    def export(self) -> None:
        self._export = True

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height.float() / height
        valid_ratio_w = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor] | None,
        pos_embeds: list[Tensor],
        refpoint_embed: Tensor,
        query_feat: Tensor,
        cross_attn_srcs: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor | None, ...]:
        src_flatten = []
        mask_flatten_parts: list[Tensor] | None = [] if masks is not None else None
        lvl_pos_embed_flatten_parts = []
        spatial_shapes_hw: list[tuple[int, int]] = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            _, c, h, w = src.shape
            spatial_shapes_hw.append((h, w))

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed_flatten_parts.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)  # bs, hw
                assert mask_flatten_parts is not None
                mask_flatten_parts.append(mask)

        memory = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten: Tensor | None = None
        valid_ratios: Tensor | None = None
        if masks is not None:
            assert mask_flatten_parts is not None
            mask_flatten = torch.cat(mask_flatten_parts, 1)  # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten_parts, 1)  # bs, \sum{hxw}, c
        # spatial_shapes must not be built by torch.empty(...) + in-place index assignment:
        # that emits a ScatterND feeding a shape tensor (level_start_index), which TensorRT
        # rejects ("IScatterLayer cannot be used to compute a shape tensor").
        # torch.as_tensor(python-int list) avoids ScatterND but bakes values as a Constant.
        # torch.stack of per-level torch._shape_as_tensor slices also produces a Constant
        # node in TorchScript ONNX export (the tracer records concrete H,W values at trace
        # time), but that Constant is accepted by TensorRT as a valid shape tensor source —
        # unlike ScatterND. torch._shape_as_tensor(t) is a private ATen op that returns a
        # 1-D int64 tensor of t's dimension sizes; [2:4] extracts (H, W) from NCHW.
        spatial_shapes = torch.stack([torch._shape_as_tensor(src)[2:4] for src in srcs]).to(
            device=srcs[0].device, dtype=torch.long
        )
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Flatten optional dual-projector features for keypoint-specific cross-attention.
        cross_attn_memory = None
        if cross_attn_srcs is not None:
            ca_flatten = []
            for cross_src in cross_attn_srcs:
                tensor = getattr(cross_src, "tensors", cross_src)
                ca_flatten.append(tensor.flatten(2).transpose(1, 2))
            cross_attn_memory = torch.cat(ca_flatten, 1)

        if self.two_stage:
            assert self.enc_out_class_embed is not None
            assert self.enc_out_bbox_embed is not None
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes_hw, unsigmoid=not self.bbox_reparam
            )
            # group detr for first stage
            refpoint_embed_ts_parts, memory_ts_parts, boxes_ts_parts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))

                enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)
                    enc_outputs_coord_cxcy_gidx = (
                        enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    )
                    enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                    enc_outputs_coord_unselected_gidx = torch.concat(
                        [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                    )
                else:
                    enc_outputs_coord_unselected_gidx = (
                        self.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals
                    )

                topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]  # bs, nq

                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)
                )  # unsigmoid
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model)
                )

                refpoint_embed_ts_parts.append(refpoint_embed_gidx)
                memory_ts_parts.append(tgt_undetach_gidx)
                boxes_ts_parts.append(refpoint_embed_gidx_undetach)
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts = torch.cat(refpoint_embed_ts_parts, dim=1)
            # (bs, nq, d)
            memory_ts = torch.cat(memory_ts_parts, dim=1)
            boxes_ts = torch.cat(boxes_ts_parts, dim=1)

        enc_kp_predictions = None
        init_kp_ref_xy = None
        keypoint_memory_ts = None
        if self.two_stage and self.use_grouppose_keypoints and hasattr(self, "keypoint_query_initializer"):
            assert self.enc_out_keypoint_embed is not None
            batch_size, _, _ = memory_ts.shape
            keypoint_memory_ts = self.keypoint_query_initializer_enc(memory_ts)
            boxes_ref = boxes_ts if self.bbox_reparam else boxes_ts.sigmoid()
            group_detr = len(self.enc_out_keypoint_embed) if self.training else 1

            kp_mem_chunks = keypoint_memory_ts.chunk(group_detr, dim=1)
            boxes_chunks = boxes_ref.chunk(group_detr, dim=1)
            kp_pred_chunks = []
            for g_idx in range(group_detr):
                kp_delta = self.enc_out_keypoint_embed[g_idx](kp_mem_chunks[g_idx])
                ref_wh = boxes_chunks[g_idx][..., 2:].unsqueeze(-2)
                ref_xy = boxes_chunks[g_idx][..., :2].unsqueeze(-2)
                kp_xy = kp_delta[..., :2] * ref_wh + ref_xy
                kp_pred_chunks.append(torch.cat([kp_xy, kp_delta[..., 2:]], dim=-1))

            enc_kp_predictions = torch.cat(kp_pred_chunks, dim=1)
            init_kp_ref_xy = enc_kp_predictions[..., :2].detach()

        if self.dec_layers > 0:
            # Use memory.shape[0] (symbolic) instead of a Python-int `bs` constant.
            bs = memory.shape[0]
            tgt = query_feat.unsqueeze(0).expand(bs, -1, -1).contiguous()
            refpoint_embed = refpoint_embed.unsqueeze(0).expand(bs, -1, -1).contiguous()
            if self.two_stage:
                ts_len = refpoint_embed_ts.shape[-2]
                refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
                refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                    refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                    refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
                else:
                    refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

                refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

            # Insert register tokens per group
            original_num_queries_per_group = None
            if self.num_registers > 0:
                group_count = self.group_detr if self.training else 1
                original_num_queries_per_group = tgt.shape[1] // group_count
                reg_tgt = self.register_tokens.unsqueeze(0).expand(bs, -1, -1)
                reg_ref = self.register_ref_points.unsqueeze(0).expand(bs, -1, -1)
                tgt_chunks = list(tgt.split(original_num_queries_per_group, dim=1))  # type: ignore[no-untyped-call]
                ref_chunks = list(
                    refpoint_embed.split(original_num_queries_per_group, dim=1)  # type: ignore[no-untyped-call]
                )
                tgt = torch.cat([torch.cat([chunk, reg_tgt], dim=1) for chunk in tgt_chunks], dim=1)
                refpoint_embed = torch.cat([torch.cat([chunk, reg_ref], dim=1) for chunk in ref_chunks], dim=1)
                if init_kp_ref_xy is not None:
                    num_keypoints = init_kp_ref_xy.shape[2]
                    reg_kp_xy = self.register_ref_points[:, :2].sigmoid()
                    reg_kp_xy = reg_kp_xy.unsqueeze(0).unsqueeze(2).expand(bs, -1, num_keypoints, -1)
                    kp_ref_chunks = list(
                        init_kp_ref_xy.split(original_num_queries_per_group, dim=1)  # type: ignore[no-untyped-call]
                    )
                    init_kp_ref_xy = torch.cat([torch.cat([chunk, reg_kp_xy], dim=1) for chunk in kp_ref_chunks], dim=1)

            tgt_keypoints = None
            if self.use_grouppose_keypoints:
                if not hasattr(self, "keypoint_query_initializer"):
                    raise ValueError("use_grouppose_keypoints=True requires keypoint initializers")
                tgt_keypoints = self.keypoint_query_initializer(tgt)

            # Route memories: kp_only mode keeps main features for detection and
            # second projector memory for keypoint cross-attention.
            if self.dual_projector_kp_only and cross_attn_memory is not None:
                decoder_memory = memory
                kp_cross_attn_memory = cross_attn_memory
            else:
                decoder_memory = cross_attn_memory if cross_attn_memory is not None else memory
                kp_cross_attn_memory = None

            decoder_outputs = self.decoder(
                tgt,
                decoder_memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.to(decoder_memory.dtype) if valid_ratios is not None else valid_ratios,
                tgt_keypoints=tgt_keypoints,
                init_kp_ref_xy=init_kp_ref_xy,
                kp_cross_attn_memory=kp_cross_attn_memory,
            )

            if self.use_grouppose_keypoints and len(decoder_outputs) > 2:
                hs, references, keypoint_hs = decoder_outputs[:3]
            else:
                hs, references = decoder_outputs[:2]
                keypoint_hs = None

            # Remove register tokens from decoder outputs.
            if self.num_registers > 0 and original_num_queries_per_group is not None:
                group_count = self.group_detr if self.training else 1
                n_with_reg = hs.shape[2] // group_count
                hs = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in hs.split(n_with_reg, dim=2)],
                    dim=2,
                )
                references = torch.cat(
                    [c[:, :, :original_num_queries_per_group, :] for c in references.split(n_with_reg, dim=2)],
                    dim=2,
                )
                if keypoint_hs is not None:
                    keypoint_hs = torch.cat(
                        [c[:, :, :original_num_queries_per_group] for c in keypoint_hs.split(n_with_reg, dim=2)],
                        dim=2,
                    )
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            hs = None
            references = None
            keypoint_hs = None

        return_values = [hs, references]
        if self.two_stage:
            return_values.append(memory_ts)
            if self.bbox_reparam:
                return_values.append(boxes_ts)
            else:
                return_values.append(boxes_ts.sigmoid())
        else:
            return_values.extend([None, None])

        if self.use_grouppose_keypoints:
            return_values.append(keypoint_hs)
            return_values.append(enc_kp_predictions)
            return_values.append(keypoint_memory_ts if self.two_stage else None)

        return tuple(return_values)


class TransformerDecoder(nn.Module):
    """Decoder stack used by DETR transformer."""

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
        d_model: int = 256,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
        enable_keypoint_processing: bool = False,
        num_keypoints_per_class: list[int] | None = None,
        grouppose_keypoint_dim_downscale: int = 1,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam
        self.enable_keypoint_processing = enable_keypoint_processing
        self.num_keypoints_per_class = num_keypoints_per_class
        self.grouppose_keypoint_dim_downscale = grouppose_keypoint_dim_downscale
        # Populated externally (e.g. by LWDETR) when iterative bbox refinement is active.
        # Declared here so that ``hasattr(self, "bbox_embed")`` short-circuits even without an
        # external injection, and so that mypy sees a stable attribute type.
        self.bbox_embed: nn.Module | None = None

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.keypoint_pos_embed = None
        if enable_keypoint_processing and num_keypoints_per_class:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            self.keypoint_pos_embed = nn.Parameter(torch.randn(sum(num_keypoints_per_class), kp_dim))
            self._create_keypoint_class_mask()
        self._export = False

    def export(self) -> None:
        self._export = True

    def _create_keypoint_class_mask(self) -> Tensor:
        """Create attention mask that blocks cross-class keypoint interactions."""
        # NOTE: near-duplicate of LWDETR._create_keypoint_class_mask in models/lwdetr.py (same
        # mask logic; that one is a pure static taking num_keypoints_per_class, this reads
        # self.num_keypoints_per_class and registers the keypoint_class_mask buffer). Keep in sync.
        if not self.num_keypoints_per_class:
            mask = torch.zeros(1, 1, dtype=torch.bool)
        else:
            total_kp = sum(self.num_keypoints_per_class)
            mask = torch.zeros(1 + total_kp, 1 + total_kp, dtype=torch.bool)
            offset = 1
            for class_idx_i, num_kp_i in enumerate(self.num_keypoints_per_class):
                if num_kp_i == 0:
                    continue
                start_i = offset + sum(self.num_keypoints_per_class[:class_idx_i])
                end_i = start_i + num_kp_i
                for class_idx_j, num_kp_j in enumerate(self.num_keypoints_per_class):
                    if num_kp_j == 0 or class_idx_i == class_idx_j:
                        continue
                    start_j = offset + sum(self.num_keypoints_per_class[:class_idx_j])
                    end_j = start_j + num_kp_j
                    mask[start_i:end_i, start_j:end_j] = True

        if "keypoint_class_mask" in self._buffers:
            self._buffers["keypoint_class_mask"] = mask
        else:
            self.register_buffer("keypoint_class_mask", mask, persistent=True)
        return cast(Tensor, self.keypoint_class_mask)

    def refpoints_refine(self, refpoints_unsigmoid: Tensor, new_refpoints_delta: Tensor) -> Tensor:
        if self.bbox_reparam:
            new_refpoints_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            )
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_unsigmoid = torch.concat([new_refpoints_cxcy, new_refpoints_wh], dim=-1)
        else:
            new_refpoints_unsigmoid = refpoints_unsigmoid + new_refpoints_delta
        return new_refpoints_unsigmoid

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        refpoints_unsigmoid: Tensor | None = None,
        # for memory
        level_start_index: Tensor | None = None,  # num_levels
        spatial_shapes: Tensor | None = None,  # num_levels, 2
        valid_ratios: Tensor | None = None,
        # keypoints
        tgt_keypoints: Tensor | None = None,
        init_kp_ref_xy: Tensor | None = None,
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        assert refpoints_unsigmoid is not None
        output = tgt

        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        keypoint_tgt = None
        kp_query_pos = None
        intermediate_keypoints = []

        if self.enable_keypoint_processing:
            assert self.lite_refpoint_refine, "Keypoint processing requires lite_refpoint_refine"
            if tgt_keypoints is None:
                raise ValueError("Keypoint processing is enabled but tgt_keypoints was not provided")
            if init_kp_ref_xy is None:
                raise ValueError("Keypoint processing is enabled but init_kp_ref_xy was not provided")
            keypoint_tgt = tgt_keypoints
            assert self.keypoint_pos_embed is not None, "keypoint_pos_embed must be initialized for keypoint processing"
            kp_query_pos = (
                self.keypoint_pos_embed.unsqueeze(0)
                .unsqueeze(0)
                .expand(keypoint_tgt.shape[0], keypoint_tgt.shape[1], -1, -1)
            )

        def get_reference(refpoints: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            # [num_queries, batch_size, 4]
            obj_center = refpoints[..., :4]

            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model // 2)  # bs, nq, 256*2
                refpoints_input = obj_center[:, :, None]  # bs, nq, 1, 4
            else:
                assert valid_ratios is not None
                refpoints_input = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                query_sine_embed = gen_sineembed_for_position(refpoints_input[:, :, 0, :], self.d_model // 2)

            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, _query_sine_embed = get_reference(
                        refpoints_unsigmoid.sigmoid()
                    )

            if self.enable_keypoint_processing and keypoint_tgt is not None:
                layer_outputs = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    query_pos=query_pos,
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    keypoint_tgt=keypoint_tgt,
                    keypoint_pos=kp_query_pos,
                    keypoint_class_mask=self.keypoint_class_mask,
                    kp_cross_attn_memory=kp_cross_attn_memory,
                )
                output, keypoint_tgt = layer_outputs
                intermediate_keypoints.append(keypoint_tgt)
            else:
                output = layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    query_pos=query_pos,
                    reference_points=refpoints_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )

            if not self.lite_refpoint_refine:
                assert self.bbox_embed is not None
                new_refpoints_delta = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                assert self.norm is not None
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                hs = intermediate[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                else:
                    ref = refpoints_unsigmoid

                if self.enable_keypoint_processing:
                    return hs, ref, intermediate_keypoints[-1]
                return hs, ref

            results = []
            if self.bbox_embed is not None:
                results.append(torch.stack(intermediate))
                results.append(torch.stack(hs_refpoints_unsigmoid))
            else:
                results.append(torch.stack(intermediate))
                results.append(refpoints_unsigmoid.unsqueeze(0))

            if self.enable_keypoint_processing:
                results.append(torch.stack(intermediate_keypoints))

            return tuple(results)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    """A single decoder layer with optional keypoint subnetwork."""

    def __init__(
        self,
        d_model: int,
        sa_nhead: int,
        ca_nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        group_detr: int = 1,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        skip_self_attn: bool = False,
        enable_keypoint_processing: bool = False,
        grouppose_keypoint_dim_downscale: int = 1,
        keypoint_cross_attn: bool = True,
        inter_instance_kp_attn: bool = False,
    ) -> None:
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = MSDeformAttn(
            d_model,
            n_levels=num_feature_levels,
            n_heads=ca_nhead,
            n_points=dec_n_points,
        )

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

        self.enable_keypoint_processing = enable_keypoint_processing
        self.inter_instance_kp_attn = inter_instance_kp_attn and enable_keypoint_processing
        self.keypoint_cross_attn = keypoint_cross_attn and enable_keypoint_processing

        if enable_keypoint_processing:
            kp_dim = d_model // grouppose_keypoint_dim_downscale
            self.inst_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.inst_pos_in_proj = (
                nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            )
            self.inst_out_proj = nn.Linear(kp_dim, d_model) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.memory_in_proj = nn.Linear(d_model, kp_dim) if grouppose_keypoint_dim_downscale > 1 else nn.Identity()
            self.kp_inst_self_attn = nn.MultiheadAttention(
                embed_dim=kp_dim,
                num_heads=_safe_multinormalize(sa_nhead // grouppose_keypoint_dim_downscale),
                dropout=dropout,
                batch_first=True,
            )
            self.kp_inst_dropout = nn.Dropout(dropout)
            self.kp_inst_norm = nn.LayerNorm(d_model)
            self.kp_norm = nn.LayerNorm(kp_dim)
            self.kp_dropout = nn.Dropout(dropout)

            if self.inter_instance_kp_attn:
                self.inter_inst_kp_attn = nn.MultiheadAttention(
                    embed_dim=kp_dim,
                    num_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    dropout=dropout,
                    batch_first=True,
                )
                self.inter_inst_kp_dropout = nn.Dropout(dropout)
                self.inter_inst_kp_norm = nn.LayerNorm(kp_dim)

            if self.keypoint_cross_attn:
                self.kp_cross_attn = MSDeformAttn(
                    kp_dim,
                    n_levels=num_feature_levels,
                    n_heads=_safe_multinormalize(ca_nhead // grouppose_keypoint_dim_downscale),
                    n_points=dec_n_points,
                )
                self.kp_cross_attn_dropout = nn.Dropout(dropout)
                self.kp_cross_attn_norm = nn.LayerNorm(kp_dim)

            self.kp_linear1 = nn.Linear(kp_dim, d_model * 4 // grouppose_keypoint_dim_downscale)
            self.kp_dropout2 = nn.Dropout(dropout)
            self.kp_linear3 = nn.Linear(d_model * 4 // grouppose_keypoint_dim_downscale, kp_dim)
            self.kp_dropout4 = nn.Dropout(dropout)
            self.kp_norm5 = nn.LayerNorm(kp_dim)

            self.instance_kp_layer_scale = nn.Parameter(torch.ones(1) * 1e-6)
        self._export = False

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        # Keypoint processing parameters
        keypoint_tgt: Tensor | None = None,  # [B, N, total_kp_per_instance, C]
        keypoint_pos: Tensor | None = None,  # [B, N, total_kp_per_instance, C]
        keypoint_class_mask: Tensor | None = None,  # [1 + K, 1 + K]
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        bs, num_queries, _ = tgt.shape

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)  # type: ignore[no-untyped-call]

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]

        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ========== End of Self-Attention =============

        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.enable_keypoint_processing:
            if keypoint_tgt is None or keypoint_pos is None:
                raise ValueError("Keypoint processing is enabled but keypoint_tgt/keypoint_pos missing")
            if reference_points is None:
                raise ValueError("Keypoint processing is enabled but reference_points missing")

            tgt_for_kp = self.inst_in_proj(tgt)
            tgt_for_kp_pos = self.inst_pos_in_proj(query_pos)

            # ========== Begin of Keypoint-Instance Self-Attention =============
            _, n_queries, num_kp, kp_dim = keypoint_tgt.shape

            tgt_expanded = tgt_for_kp.unsqueeze(2)  # [B, N, 1, C]
            query_expanded = torch.zeros_like(tgt_for_kp).unsqueeze(2)  # [B, N, 1, C]

            combined_feat = torch.cat([tgt_expanded, keypoint_tgt], dim=2)  # [B, N, 1 + K, C]
            combined_pos = torch.cat([query_expanded, keypoint_pos], dim=2)  # [B, N, 1 + K, C]

            combined_feat = combined_feat.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            combined_pos = combined_pos.reshape(bs * num_queries, 1 + num_kp, kp_dim)
            q = k = combined_feat + combined_pos
            v = combined_feat

            combined_out = self.kp_inst_self_attn(q, k, v, attn_mask=keypoint_class_mask, need_weights=False)[0]
            combined_out = combined_out.reshape(bs, num_queries, 1 + num_kp, kp_dim)
            tgt2 = combined_out[:, :, 0, :]
            keypoint_tgt2 = combined_out[:, :, 1:, :]

            tgt = tgt + self.kp_inst_dropout(self.inst_out_proj(tgt2)) * self.instance_kp_layer_scale
            tgt = self.kp_inst_norm(tgt)
            keypoint_tgt = keypoint_tgt + self.kp_dropout(keypoint_tgt2)
            keypoint_tgt = self.kp_norm(keypoint_tgt)

            # ========== End of Keypoint-Instance Self-Attention =============

            # ========== Begin of Cross-Keypoint Attention =============
            if self.inter_instance_kp_attn:
                swapped_keypoint_tgt = keypoint_tgt.transpose(1, 2).reshape(bs * num_kp, num_queries, kp_dim)
                swapped_keypoint_pos = (
                    tgt_for_kp_pos.unsqueeze(1)
                    .expand(bs, num_kp, num_queries, kp_dim)
                    .reshape(
                        bs * num_kp,
                        num_queries,
                        kp_dim,
                    )
                )
                q = swapped_keypoint_tgt + swapped_keypoint_pos
                v = swapped_keypoint_tgt
                swapped_out = self.inter_inst_kp_attn(q, key=q, value=v, need_weights=False)[0]
                swapped_out = swapped_out.view(bs, num_kp, num_queries, kp_dim).transpose(1, 2)
                keypoint_tgt = keypoint_tgt + self.inter_inst_kp_dropout(swapped_out)
                keypoint_tgt = self.inter_inst_kp_norm(keypoint_tgt)

            # ========== End of Cross-Keypoint Attention =============

            # ========== Begin of Keypoint-Specific Cross-Attention =============
            if self.keypoint_cross_attn:
                keypoint_query = self.with_pos_embed(
                    keypoint_tgt, tgt_for_kp_pos.unsqueeze(2).expand(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_query = keypoint_query.reshape(bs, num_queries * num_kp, kp_dim)
                bbox_ref_for_kp = (
                    reference_points.unsqueeze(2)
                    .expand(
                        bs,
                        num_queries,
                        num_kp,
                        reference_points.shape[2],
                        reference_points.shape[3],
                    )
                    .reshape(bs, num_queries * num_kp, reference_points.shape[2], reference_points.shape[3])
                )
                kp_memory = kp_cross_attn_memory if kp_cross_attn_memory is not None else memory
                keypoint_tgt = keypoint_tgt + self.kp_cross_attn_dropout(
                    self.kp_cross_attn(
                        keypoint_query,
                        bbox_ref_for_kp,
                        self.memory_in_proj(kp_memory),
                        spatial_shapes,
                        level_start_index,
                        memory_key_padding_mask,
                    ).reshape(bs, num_queries, num_kp, kp_dim)
                )
                keypoint_tgt = self.kp_cross_attn_norm(keypoint_tgt)

            # ========== End of Keypoint-Specific Cross-Attention =============

            # ========== Begin of Keypoint-Specific FFN =============
            keypoint_tgt = keypoint_tgt + self.kp_dropout4(
                self.kp_linear3(self.kp_dropout2(self.activation(self.kp_linear1(keypoint_tgt))))
            )
            keypoint_tgt = self.kp_norm5(keypoint_tgt)
            # ========== End of Keypoint-Specific FFN =============

            return tgt, keypoint_tgt

        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        keypoint_tgt: Tensor | None = None,
        keypoint_pos: Tensor | None = None,
        keypoint_class_mask: Tensor | None = None,
        kp_cross_attn_memory: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        return self.forward_post(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            keypoint_tgt=keypoint_tgt,
            keypoint_pos=keypoint_pos,
            keypoint_class_mask=keypoint_class_mask,
            kp_cross_attn_memory=kp_cross_attn_memory,
        )


def _get_clones(module: nn.Module, num_clones: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_clones)])


def build_transformer(args: BuilderArgs) -> Transformer:
    two_stage = getattr(args, "two_stage", False)

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,  # type: ignore[attr-defined]
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,  # type: ignore[attr-defined]
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,  # type: ignore[attr-defined]
        bbox_reparam=args.bbox_reparam,
        # Detection-only builder args may omit keypoint-only fields; default to the non-keypoint path.
        use_grouppose_keypoints=getattr(args, "use_grouppose_keypoints", False),
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        grouppose_keypoint_dim_downscale=getattr(args, "grouppose_keypoint_dim_downscale", 1),
        keypoint_cross_attn=getattr(args, "keypoint_cross_attn", True),
        inter_instance_kp_attn=getattr(args, "inter_instance_kp_attn", False),
        num_registers=getattr(args, "num_decoder_registers", 0),
        dual_projector_kp_only=getattr(args, "dual_projector_kp_only", False),
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

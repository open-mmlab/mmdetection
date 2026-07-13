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
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""LW-DETR model and criterion classes."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from mmdet.models.layers.rf_detr.config import ModelConfig, TrainConfig

from mmdet.models.layers.rf_detr.models._defaults import MODEL_DEFAULTS, ModelDefaults
from mmdet.models.layers.rf_detr.models._types import BuilderArgs
from mmdet.models.layers.rf_detr.models.backbone import Joiner, build_backbone

# Backward-compat re-exports: loss functions that used to live in this module
from mmdet.models.layers.rf_detr.models.criterion import (  # noqa: F401 — backward compat
    SetCriterion,
    dice_loss,
    dice_loss_jit,
    position_supervised_loss,
    sigmoid_ce_loss,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
)
from mmdet.models.layers.rf_detr.models.heads.segmentation import SegmentationHead
from mmdet.models.layers.rf_detr.models.matcher import build_matcher
from mmdet.models.layers.rf_detr.models.math import MLP
from mmdet.models.layers.rf_detr.models.postprocess import PostProcess
from mmdet.models.layers.rf_detr.models.transformer import Transformer, build_transformer
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

logger = get_logger()


def _resize_linear(linear: nn.Linear, num_classes: int) -> nn.Linear:
    """Return a new :class:`~torch.nn.Linear` resized to *num_classes* outputs.

    Tiles the existing weight rows when *num_classes* is larger than the current output size, or truncates them when
    smaller.  The returned module has ``out_features == num_classes`` so that ``nn.Linear`` metadata stays consistent
    with the actual weight shape — a requirement for correct ONNX export and ``torch.jit.trace`` serialisation.

    Args:
        linear: Source linear layer whose weights are used as the starting point.
        num_classes: Target number of output features.

    Returns:
        A new :class:`~torch.nn.Linear` with ``in_features`` unchanged and ``out_features == num_classes``.
    """
    base = linear.weight.shape[0]
    num_repeats = int(math.ceil(num_classes / base))
    new_weight = linear.weight.detach().repeat(num_repeats, 1)[:num_classes]
    new_bias = linear.bias.detach().repeat(num_repeats)[:num_classes] if linear.bias is not None else None
    new_linear = nn.Linear(linear.in_features, num_classes, bias=new_bias is not None)
    # Copy resized weights/bias into the new layer while preserving requires_grad flags.
    with torch.no_grad():
        new_linear.weight.copy_(new_weight)
        if new_bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_bias)
    new_linear.weight.requires_grad = linear.weight.requires_grad
    if linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.requires_grad = linear.bias.requires_grad
    return new_linear


def _resize_parameter_rows(parameter: nn.Parameter, num_rows: int) -> nn.Parameter:
    """Return a parameter with the first dimension resized by tiling or truncating rows."""
    current_rows = parameter.shape[0]
    if current_rows == num_rows:
        return parameter

    if current_rows == 0:
        new_data = parameter.detach().new_zeros((num_rows, *parameter.shape[1:]))
    else:
        repeats = [int(math.ceil(num_rows / current_rows)), *([1] * (parameter.dim() - 1))]
        new_data = parameter.detach().repeat(*repeats)[:num_rows]
    return nn.Parameter(new_data.clone(), requires_grad=parameter.requires_grad)


def _reset_keypoint_gaussian_output_rows(module: nn.Module) -> None:
    """Reset keypoint precision-Cholesky output rows to unit Gaussian values."""
    layers = getattr(module, "layers", None)
    if layers is None or len(layers) == 0:
        return

    final_layer = layers[-1]
    if not isinstance(final_layer, nn.Linear) or final_layer.out_features <= 6:
        return

    with torch.no_grad():
        final_layer.weight[4:7].zero_()
        if final_layer.bias is not None:
            final_layer.bias[4:7].zero_()


class LWDETR(nn.Module):
    """This is the Group DETR v3 module that performs object detection."""

    def __init__(
        self,
        backbone: Joiner,
        transformer: Transformer,
        segmentation_head: SegmentationHead | None,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
        use_grouppose_keypoints: bool = False,
        num_keypoints_per_class: list[int] | None = None,
        grouppose_keypoint_dim_downscale: int = 1,
    ) -> None:
        """Initializes the model.

        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head
        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # GroupPose keypoint heads (group_detr style) — only if GroupPose keypoint mode is active.
        self.use_grouppose_keypoints = use_grouppose_keypoints
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.grouppose_keypoint_dim_downscale = grouppose_keypoint_dim_downscale
        if self.use_grouppose_keypoints and len(self.num_keypoints_per_class) > num_classes:
            raise ValueError(
                f"num_keypoints_per_class has {len(self.num_keypoints_per_class)} entries but the detection head "
                f"only has {num_classes} classes. Class-logit boosts for keypoint classes with id >= {num_classes} "
                "would be silently truncated. Increase num_classes or shorten num_keypoints_per_class."
            )
        # Flag to ensure the zero-pad warning in ``_aggregate_keypoint_class_logits`` is emitted at most once.
        self._kp_zero_pad_warned = False
        self.keypoint_embed: MLP | None
        if self.use_grouppose_keypoints:
            self.keypoint_embed = MLP(
                hidden_dim // self.grouppose_keypoint_dim_downscale,
                hidden_dim // self.grouppose_keypoint_dim_downscale,
                8,
                3,
            )
            # Initialize keypoint head to near-identity behavior around zero delta.
            keypoint_out_layer = cast(nn.Linear, self.keypoint_embed.layers[-1])
            nn.init.constant_(keypoint_out_layer.weight.data, 0)
            nn.init.constant_(keypoint_out_layer.bias.data, 0)
        else:
            self.keypoint_embed = None

        self._kp_active_mask: Tensor
        self.register_buffer("_kp_active_mask", self._create_kp_active_mask(self.num_keypoints_per_class))

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        bbox_out_layer = cast(nn.Linear, self.bbox_embed.layers[-1])
        nn.init.constant_(bbox_out_layer.weight.data, 0)
        nn.init.constant_(bbox_out_layer.bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )
            if self.use_grouppose_keypoints and self.keypoint_embed is not None:
                self.transformer.enc_out_keypoint_embed = nn.ModuleList(
                    [copy.deepcopy(self.keypoint_embed) for _ in range(group_detr)]
                )

        self._export = False

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Resize the detection classification head to *num_classes* outputs.

        Replaces ``self.class_embed`` (and each ``enc_out_class_embed`` when the model uses two-stage detection) with a
        new :class:`torch.nn.Linear` whose ``out_features`` equals *num_classes*.  When *num_classes* is larger than the
        current head the existing weights are tiled; when smaller they are truncated.  Replacing the module (rather than
        mutating ``.data``) keeps ``nn.Linear.out_features`` consistent with the actual weight shape, which is required
        for correct ONNX export.

        Args:
            num_classes: Target number of output classes (including background).
        """
        self.class_embed = _resize_linear(self.class_embed, num_classes)

        if self.two_stage:
            assert self.transformer.enc_out_class_embed is not None
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [_resize_linear(cast(nn.Linear, m), num_classes) for m in self.transformer.enc_out_class_embed]
            )

    @staticmethod
    def _create_kp_active_mask(num_keypoints_per_class: list[int]) -> Tensor:
        """Create a compact class-by-keypoint active mask for a keypoint schema."""
        if not num_keypoints_per_class:
            return torch.zeros(0, 0, dtype=torch.bool)

        max_kp = max(num_keypoints_per_class)
        kp_active = torch.zeros(len(num_keypoints_per_class), max_kp, dtype=torch.bool)
        for class_idx, num_keypoints in enumerate(num_keypoints_per_class):
            kp_active[class_idx, :num_keypoints] = True
        return kp_active

    @staticmethod
    def _create_keypoint_class_mask(num_keypoints_per_class: list[int]) -> Tensor:
        """Create an attention mask that blocks cross-class keypoint interactions."""
        # NOTE: near-duplicate of TransformerDecoder._create_keypoint_class_mask in transformer.py
        # (same mask logic; that one reads self.num_keypoints_per_class and registers a buffer,
        # this pure static returns the mask). Keep in sync. Not extracted: lwdetr imports
        # transformer, so a shared helper in transformer would invert the dependency safely but
        # the differing signatures make a shared function awkward.
        if not num_keypoints_per_class:
            return torch.zeros(1, 1, dtype=torch.bool)

        total_keypoints = sum(num_keypoints_per_class)
        mask = torch.zeros(1 + total_keypoints, 1 + total_keypoints, dtype=torch.bool)
        for class_idx_i, num_keypoints_i in enumerate(num_keypoints_per_class):
            if num_keypoints_i == 0:
                continue
            start_i = 1 + sum(num_keypoints_per_class[:class_idx_i])
            end_i = start_i + num_keypoints_i
            for class_idx_j, num_keypoints_j in enumerate(num_keypoints_per_class):
                if num_keypoints_j == 0 or class_idx_i == class_idx_j:
                    continue
                start_j = 1 + sum(num_keypoints_per_class[:class_idx_j])
                end_j = start_j + num_keypoints_j
                mask[start_i:end_i, start_j:end_j] = True
        return mask

    def get_num_keypoints_per_class(self) -> list[int]:
        """Return the current keypoint schema inferred from the active-keypoint mask."""
        return [int(num_keypoints) for num_keypoints in self._kp_active_mask.sum(dim=1).tolist()]

    @staticmethod
    def get_num_keypoints_per_class_from_checkpoint(state_dict: dict[str, Tensor]) -> list[int] | None:
        """Infer the keypoint schema stored in a checkpoint state dict."""
        active_mask = state_dict.get("_kp_active_mask")
        if not isinstance(active_mask, Tensor) or active_mask.ndim != 2:
            return None
        return [int(num_keypoints) for num_keypoints in active_mask.sum(dim=1).tolist()]

    def reinitialize_keypoint_head(self, num_keypoints_per_class: list[int] | None) -> None:
        """Resize schema-dependent GroupPose state to match ``num_keypoints_per_class``."""
        if not self.use_grouppose_keypoints or not num_keypoints_per_class:
            return

        schema = list(num_keypoints_per_class)
        total_keypoints = sum(schema)
        self.num_keypoints_per_class = schema
        self._kp_active_mask = self._create_kp_active_mask(schema).to(self._kp_active_mask.device)

        if hasattr(self.transformer, "num_keypoints_per_class"):
            self.transformer.num_keypoints_per_class = schema

        decoder = getattr(self.transformer, "decoder", None)
        if decoder is not None:
            if hasattr(decoder, "num_keypoints_per_class"):
                decoder.num_keypoints_per_class = schema
            keypoint_pos_embed = getattr(decoder, "keypoint_pos_embed", None)
            if isinstance(keypoint_pos_embed, nn.Parameter):
                decoder.keypoint_pos_embed = _resize_parameter_rows(keypoint_pos_embed, total_keypoints)
            if hasattr(decoder, "_create_keypoint_class_mask"):
                decoder._create_keypoint_class_mask()
            elif hasattr(decoder, "keypoint_class_mask"):
                current_mask = decoder.keypoint_class_mask
                decoder.keypoint_class_mask = self._create_keypoint_class_mask(schema).to(current_mask.device)

        for initializer_name in ("keypoint_query_initializer", "keypoint_query_initializer_enc"):
            initializer = getattr(self.transformer, initializer_name, None)
            queries = getattr(initializer, "queries", None)
            if initializer is not None and isinstance(queries, nn.Parameter):
                initializer.queries = _resize_parameter_rows(queries, total_keypoints)

    def reset_keypoint_gaussian_parameters(self) -> None:
        """Reset keypoint Gaussian precision outputs to unit values.

        Keypoint channels 4, 5, and 6 encode the lower-triangular precision
        Cholesky parameters ``log_l11``, ``l21``, and ``log_l22``. Zeroing the
        final prediction rows gives ``L = identity`` at the start of finetuning
        while preserving learned keypoint location, visibility, findability, and
        class-logit channels loaded from the checkpoint.
        """
        if not self.use_grouppose_keypoints or self.keypoint_embed is None:
            return

        _reset_keypoint_gaussian_output_rows(self.keypoint_embed)
        enc_keypoint_embed = getattr(self.transformer, "enc_out_keypoint_embed", None)
        if isinstance(enc_keypoint_embed, nn.ModuleList):
            for keypoint_embed in enc_keypoint_embed:
                _reset_keypoint_gaussian_output_rows(keypoint_embed)

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign,assignment]
        for name, m in self.named_modules():
            if hasattr(m, "export") and callable(m.export) and hasattr(m, "_export") and not m._export:
                m.export()

    def _format_keypoint_output(
        self,
        keypoints_compact: Tensor,
        batch_size_expected: int,
        num_queries_expected: int,
    ) -> Tensor:
        """Convert compact GroupPose keypoints to class-padded keypoint layout."""
        if not self.use_grouppose_keypoints or not self.num_keypoints_per_class:
            return keypoints_compact

        if keypoints_compact.dim() != 4:
            return keypoints_compact

        batch_size, num_queries, total_compact_keypoints, keypoint_dim = keypoints_compact.shape
        if batch_size != batch_size_expected or num_queries != num_queries_expected:
            raise ValueError(
                f"_format_keypoint_output received tensor with batch_size={batch_size}, num_queries={num_queries} "
                f"but expected batch_size={batch_size_expected}, num_queries={num_queries_expected}. "
                "Shape mismatch silently bypassed in earlier versions; raise to surface upstream bugs."
            )

        max_num_keypoints = max(self.num_keypoints_per_class)
        total_padded_keypoints = len(self.num_keypoints_per_class) * max_num_keypoints
        total_actual_keypoints = sum(self.num_keypoints_per_class)

        if total_compact_keypoints == total_padded_keypoints:
            # Already in the per-class padded layout — pass through.
            return keypoints_compact
        if total_compact_keypoints != total_actual_keypoints:
            raise ValueError(
                f"_format_keypoint_output received tensor with total_compact_keypoints={total_compact_keypoints} "
                f"but schema expects either compact total={total_actual_keypoints} "
                f"or padded total={total_padded_keypoints} for num_keypoints_per_class={self.num_keypoints_per_class}."
            )

        padded = torch.zeros(
            batch_size,
            num_queries,
            total_padded_keypoints,
            keypoint_dim,
            dtype=keypoints_compact.dtype,
            device=keypoints_compact.device,
        )
        compact_idx = 0
        for class_idx, keypoint_count in enumerate(self.num_keypoints_per_class):
            class_offset = class_idx * max_num_keypoints
            for keypoint_idx in range(keypoint_count):
                padded[:, :, class_offset + keypoint_idx, :] = keypoints_compact[:, :, compact_idx, :]
                compact_idx += 1
        return padded

    def _aggregate_keypoint_class_logits(self, keypoint_predictions: Tensor) -> Tensor:
        """Aggregate keypoint class-logit contributions into detection-class logits."""
        if not self.num_keypoints_per_class:
            return torch.zeros(
                (*keypoint_predictions.shape[:-2], self.class_embed.out_features),
                dtype=keypoint_predictions.dtype,
                device=keypoint_predictions.device,
            )

        num_keypoint_classes = len(self.num_keypoints_per_class)
        max_num_keypoints = max(self.num_keypoints_per_class)
        class_contrib = keypoint_predictions[..., 7].view(
            *keypoint_predictions.shape[:-2], num_keypoint_classes, max_num_keypoints
        )
        class_contrib = class_contrib * self._kp_active_mask.to(class_contrib.dtype)
        class_boost = class_contrib.sum(dim=-1)

        detection_num_classes = self.class_embed.out_features
        foreground_num_classes = detection_num_classes - 1
        if class_boost.shape[-1] < detection_num_classes:
            if class_boost.shape[-1] < foreground_num_classes:
                # Only warn when the schema doesn't cover all foreground detection classes.
                # The background slot (index detection_num_classes-1) always receives zero boost,
                # so a schema that covers exactly num_classes foreground classes is correct and
                # does not warrant a warning.
                if not self._kp_zero_pad_warned:
                    logger.warning(
                        "Keypoint class-logit boost has %d classes but detection head has %d foreground classes; "
                        "zero-padding boost for classes %d..%d. Detection classes with no keypoint schema "
                        "will receive zero boost. This warning is emitted once per model instance.",
                        class_boost.shape[-1],
                        foreground_num_classes,
                        class_boost.shape[-1],
                        foreground_num_classes - 1,
                    )
                    self._kp_zero_pad_warned = True
            class_boost = torch.cat(
                [
                    class_boost,
                    class_boost.new_zeros(*class_boost.shape[:-1], detection_num_classes - class_boost.shape[-1]),
                ],
                dim=-1,
            )
        elif class_boost.shape[-1] > detection_num_classes:
            # Unreachable under normal use: ``__init__`` raises ``ValueError`` when
            # ``len(num_keypoints_per_class) > num_classes``. Kept defensively in case
            # ``class_embed`` is resized post-init.
            class_boost = class_boost[..., :detection_num_classes]
        return class_boost

    def forward(self, samples: NestedTensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Any]:
        """The forward expects a NestedTensor, which consists of:

           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1], relative to the
                           size of each individual image (disregarding possible padding). See PostProcess for
                           information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss, cross_attn_features = self.backbone(samples)

        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = self.segmentation_head.sparse_forward if self.training else self.segmentation_head.forward

        cross_attn_srcs = None
        if cross_attn_features is not None:
            cross_attn_srcs = []
            for feature in cross_attn_features:
                cross_src, _ = feature.decompose()
                cross_attn_srcs.append(cross_src)

        transformer_outputs = self.transformer(
            srcs,
            masks,
            poss,
            refpoint_embed_weight,
            query_feat_weight,
            cross_attn_srcs=cross_attn_srcs,
        )
        if self.use_grouppose_keypoints:
            hs, ref_unsigmoid, hs_enc, ref_enc, keypoint_hs, enc_kp_predictions, _ = transformer_outputs
        else:
            hs, ref_unsigmoid, hs_enc, ref_enc = transformer_outputs[:4]
            keypoint_hs = None
            enc_kp_predictions = None

        out: dict[str, Any] = {}
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)
            outputs_keypoints = None

            if self.use_grouppose_keypoints and self.keypoint_embed is not None:
                if keypoint_hs is None:
                    raise ValueError("use_grouppose_keypoints=True requires keypoint_hs from transformer outputs.")

                outputs_keypoints_delta = self.keypoint_embed(keypoint_hs)
                ref_wh = ref_unsigmoid[..., 2:].unsqueeze(-2)
                ref_xy = ref_unsigmoid[..., :2].unsqueeze(-2)
                keypoints_xy = outputs_keypoints_delta[..., :2] * ref_wh + ref_xy
                keypoints_other = outputs_keypoints_delta[..., 2:]
                outputs_keypoints_compact = torch.cat([keypoints_xy, keypoints_other], dim=-1)

                layer_outputs_keypoints = []
                for layer_idx in range(outputs_keypoints_compact.shape[0]):
                    compact_preds = outputs_keypoints_compact[layer_idx]
                    layer_outputs_keypoints.append(
                        self._format_keypoint_output(
                            compact_preds,
                            compact_preds.shape[0],
                            compact_preds.shape[1],
                        )
                    )
                outputs_keypoints = torch.stack(layer_outputs_keypoints, dim=0)
                outputs_class = outputs_class + self._aggregate_keypoint_class_logits(outputs_keypoints)

            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(features[0].tensors, hs, cast(tuple[int, int], samples.tensors.shape[-2:]))

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]
            if outputs_keypoints is not None:
                out["pred_keypoints"] = outputs_keypoints[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                    outputs_keypoints,
                )

        if self.two_stage:
            assert self.transformer.enc_out_class_embed is not None
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc_list = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc_list.append(cls_enc_gidx)

            cls_enc = torch.cat(cls_enc_list, dim=1)
            keypoints_enc = None
            if self.use_grouppose_keypoints and enc_kp_predictions is not None:
                keypoints_enc = self._format_keypoint_output(
                    enc_kp_predictions,
                    enc_kp_predictions.shape[0],
                    enc_kp_predictions.shape[1],
                )
                cls_enc = cls_enc + self._aggregate_keypoint_class_logits(keypoints_enc)

            if self.segmentation_head is not None:
                masks_enc = seg_head_fwd(
                    features[0].tensors,
                    [
                        hs_enc,
                    ],
                    cast(tuple[int, int], samples.tensors.shape[-2:]),
                    skip_blocks=True,
                )[0]

            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["enc_outputs"]["pred_masks"] = masks_enc
                if keypoints_enc is not None:
                    out["enc_outputs"]["pred_keypoints"] = keypoints_enc
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["pred_masks"] = masks_enc
                if keypoints_enc is not None:
                    out["pred_keypoints"] = keypoints_enc

        return out

    def forward_export(self, tensors: Tensor) -> tuple[Tensor, ...]:
        srcs, _, poss, cross_attn_srcs = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]

        transformer_outputs = self.transformer(
            srcs,
            None,
            poss,
            refpoint_embed_weight,
            query_feat_weight,
            cross_attn_srcs=cross_attn_srcs,
        )
        if self.use_grouppose_keypoints:
            hs, ref_unsigmoid, hs_enc, ref_enc, keypoint_hs, enc_kp_predictions, _ = transformer_outputs
        else:
            hs, ref_unsigmoid, hs_enc, ref_enc = transformer_outputs[:4]
            keypoint_hs = None
            enc_kp_predictions = None

        outputs_masks = None
        outputs_keypoints = None

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.use_grouppose_keypoints and self.keypoint_embed is not None:
                if keypoint_hs is None:
                    raise ValueError("use_grouppose_keypoints=True requires keypoint_hs from transformer outputs.")
                outputs_keypoints_delta = self.keypoint_embed(keypoint_hs)
                ref_wh = ref_unsigmoid[..., 2:].unsqueeze(-2)
                ref_xy = ref_unsigmoid[..., :2].unsqueeze(-2)
                keypoints_xy = outputs_keypoints_delta[..., :2] * ref_wh + ref_xy
                keypoints_other = outputs_keypoints_delta[..., 2:]
                outputs_keypoints = torch.cat([keypoints_xy, keypoints_other], dim=-1)
                if outputs_keypoints.dim() == 5:
                    outputs_keypoints = outputs_keypoints[-1]
                outputs_keypoints = self._format_keypoint_output(
                    outputs_keypoints,
                    outputs_keypoints.shape[0],
                    outputs_keypoints.shape[1],
                )
                outputs_class = outputs_class + self._aggregate_keypoint_class_logits(outputs_keypoints)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0],
                    [
                        hs,
                    ],
                    tensors.shape[-2:],
                )[0]
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            assert self.transformer.enc_out_class_embed is not None
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.use_grouppose_keypoints and enc_kp_predictions is not None:
                outputs_keypoints = self._format_keypoint_output(
                    enc_kp_predictions,
                    enc_kp_predictions.shape[0],
                    enc_kp_predictions.shape[1],
                )
                outputs_class = outputs_class + self._aggregate_keypoint_class_logits(outputs_keypoints)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0],
                    [
                        hs_enc,
                    ],
                    tensors.shape[-2:],
                    skip_blocks=True,
                )[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        if outputs_keypoints is not None:
            return outputs_coord, outputs_class, outputs_keypoints
        else:
            return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: Tensor,
        outputs_coord: Tensor,
        outputs_masks: list[Tensor] | list[dict[str, Tensor]] | None,
        outputs_keypoints: Tensor | None = None,
    ) -> list[dict[str, Any]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        names = ["pred_logits", "pred_boxes"]
        values: list[Any] = [outputs_class[:-1], outputs_coord[:-1]]
        if outputs_masks is not None:
            names.append("pred_masks")
            values.append(outputs_masks[:-1])
        if outputs_keypoints is not None:
            names.append("pred_keypoints")
            values.append(outputs_keypoints[:-1])
        return [{name: value for name, value in zip(names, layer_values)} for layer_values in zip(*values)]

    def _get_backbone_encoder_layers(self) -> nn.ModuleList | None:
        """Resolve the list of transformer blocks/layers from backbone[0].encoder.

        Supports multiple backbone architectures:
        - encoder.blocks (standard ViT)
        - encoder.trunk.blocks (aimv2)
        - encoder.encoder.encoder.layer (HuggingFace DinoV2)

        Returns:
            List of transformer layers, or None if not found.
        """
        enc = self.backbone[0].encoder
        if hasattr(enc, "blocks"):
            return cast(nn.ModuleList, enc.blocks)
        if hasattr(enc, "trunk") and hasattr(enc.trunk, "blocks"):
            return cast(nn.ModuleList, enc.trunk.blocks)
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "encoder") and hasattr(enc.encoder.encoder, "layer"):
            return cast(nn.ModuleList, enc.encoder.encoder.layer)
        return None

    def update_drop_path(self, drop_path_rate: float, vit_encoder_num_layers: int) -> None:
        """Update drop_path rates for backbone encoder layers with linear schedule.

        Applies a linear schedule where the first layer has drop_path_rate=0 and the last layer has
        drop_path_rate=drop_path_rate. Intermediate layers are interpolated linearly.

        Args:
            drop_path_rate: Maximum drop path rate (applied to last layer).
            vit_encoder_num_layers: Number of encoder layers to update.
        """
        layers = self._get_backbone_encoder_layers()
        if layers is None:
            return
        n = min(vit_encoder_num_layers, len(layers))
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n)]
        for i in range(n):
            if hasattr(layers[i], "drop_path") and hasattr(layers[i].drop_path, "drop_prob"):
                layers[i].drop_path.drop_prob = dp_rates[i]  # type: ignore[union-attr]

    def update_dropout(self, drop_rate: float) -> None:
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


def build_model(args: "BuilderArgs") -> LWDETR | tuple[Any, None, None]:
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.num_classes + 1
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=(
            args.shape
            if hasattr(args, "shape")
            else ((args.resolution, args.resolution) if hasattr(args, "resolution") else (640, 640))
        ),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        dual_projector=args.dual_projector,
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)  # type: ignore[attr-defined]
    transformer = build_transformer(args)

    segmentation_head = (
        SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )
        if args.segmentation_head
        else None
    )

    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        # Detection-only builder args may omit keypoint-only fields; default to the non-keypoint path.
        use_grouppose_keypoints=getattr(args, "use_grouppose_keypoints", False),
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        grouppose_keypoint_dim_downscale=getattr(args, "grouppose_keypoint_dim_downscale", 1),
    )
    return model


def build_criterion_and_postprocessors(args: "BuilderArgs") -> tuple[SetCriterion, PostProcess]:
    device = torch.device(args.device)
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.segmentation_head:
        weight_dict["loss_mask_ce"] = args.mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = args.mask_dice_loss_coef
    # Detection-only training args may omit keypoint loss knobs; read them only for keypoint mode.
    has_keypoints = getattr(args, "use_grouppose_keypoints", False)
    if has_keypoints:
        weight_dict["loss_keypoints_l1"] = getattr(args, "keypoint_l1_loss_coef", 0.0)
        weight_dict["loss_keypoints_findable"] = getattr(args, "keypoint_findable_loss_coef", 0.0)
        weight_dict["loss_keypoints_visible"] = getattr(args, "keypoint_visible_loss_coef", 0.0)
        weight_dict["loss_keypoints_nll"] = getattr(args, "keypoint_nll_loss_coef", 0.0)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.segmentation_head:
        losses.append("masks")
    if has_keypoints:
        losses.append("keypoints")

    sum_group_losses = getattr(args, "sum_group_losses", False)
    if args.segmentation_head:
        criterion = SetCriterion(
            args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
            num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        )
    else:
        criterion = SetCriterion(
            args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss,
            num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        )
    criterion.to(device)
    postprocess = PostProcess(
        num_select=args.num_select,
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        # Older detection-only namespaces may omit keypoint postprocess knobs; keep the ModelConfig default.
        trace_alpha=getattr(args, "postprocess_trace_alpha", 0.2),
    )

    return criterion, postprocess


def build_model_from_config(
    model_config: "ModelConfig",
    train_config: "TrainConfig" | None = None,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> LWDETR:
    """Build an LWDETR model directly from a ModelConfig.

    A config-native alternative to ``build_model(build_namespace(mc, tc))``. Constructs the namespace internally from
    ``model_config``, an optional ``train_config``, and ``defaults``, then delegates to :func:`build_model`.

    Note:
        The internal ``SimpleNamespace`` bridge is transitional — it will be eliminated once all builder functions
        accept config objects directly. Callers should not rely on the namespace shape or pass it externally.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration. If ``None``,
            a minimal dummy ``TrainConfig(dataset_dir=".", output_dir=".")`` is constructed, matching the previous
            default behavior.
        defaults: Hardcoded architectural constants. Defaults to ``MODEL_DEFAULTS``.

    Returns:
        Fully initialised LWDETR model.

    Raises:
        ValueError: If ``defaults`` request ``encoder_only`` or ``backbone_only``,
            which would make the return type differ from ``LWDETR``.
    """
    from mmdet.models.layers.rf_detr._namespace import _namespace_from_configs

    if defaults.encoder_only or defaults.backbone_only:
        raise ValueError(
            "build_model_from_config() requires defaults.encoder_only=False and defaults.backbone_only=False."
        )

    if train_config is None:
        from mmdet.models.layers.rf_detr.config import TrainConfig

        train_config = TrainConfig(dataset_dir=".", output_dir=".")

    ns = _namespace_from_configs(model_config, train_config, defaults)
    model = build_model(ns)
    assert isinstance(model, LWDETR), (
        "build_model() returned a non-LWDETR result even though encoder_only/backbone_only were False."
    )
    return model


def build_criterion_from_config(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> tuple[SetCriterion, PostProcess]:
    """Build criterion and postprocessor directly from config objects.

    A config-native alternative to ``build_criterion_and_postprocessors(build_namespace(mc, tc))``.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants. Defaults to ``MODEL_DEFAULTS``.

    Returns:
        A 2-tuple of ``(SetCriterion, PostProcess)``.
    """
    from mmdet.models.layers.rf_detr._namespace import _namespace_from_configs

    ns = _namespace_from_configs(model_config, train_config, defaults)
    return build_criterion_and_postprocessors(ns)

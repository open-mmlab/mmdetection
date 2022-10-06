# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         MultiScaleDeformableAttention,
                                         build_norm_layer)
from mmengine.model import ModuleList, xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from ..layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                      DetrTransformerEncoder, DetrTransformerEncoderLayer,
                      SinePositionalEncoding, inverse_sigmoid)
from .base_detr import TransformerDetector


@MODELS.register_module()
class DeformableDETR(TransformerDetector):
    r"""Implementation of `Deformable DETR: Deformable Transformers for
    End-to-End Object Detection <https://arxiv.org/abs/2010.04159>`_

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    Args:
        with_box_refine (bool, optional): Whether to refine the reference
            points in the decoder. Defaults to False.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to False.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 4.
    """

    def __init__(self,
                 *args,
                 decoder_cfg: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 **kwargs) -> None:
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            bbox_head.update(
                dict(
                    with_box_refine=with_box_refine,
                    as_two_stage=as_two_stage,
                    num_decoder_layers=decoder_cfg['num_layers']))

        super().__init__(
            *args, decoder_cfg=decoder_cfg, bbox_head=bbox_head, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()  # TODO: should this be in the front ?
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        # TODO: Doc
        """Prepare the inputs of the Transformer.

        Args:
            mlvl_feats (Tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            Dict[str, Tensor]: The inputs of the Transformer.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary mask of feat
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat, embed_dims) -> (num_feat, bs, embed_dims)
        feat_flatten = feat_flatten.permute(1, 0, 2)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        encoder_reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=encoder_reference_points)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat, feat_mask, feat_pos, spatial_shapes,
                        level_start_index, valid_ratios,
                        reference_points) -> Dict:  # TODO: typehint
        # TODO: Doc
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            query_key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        # [] -> []  # TODO
        memory = memory.permute(1, 0, 2)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        # TODO: Doc
        batch_size, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], self.num_query, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        # [] -> []  # TODO
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class if self.as_two_stage else None,
            enc_outputs_coord=enc_outputs_coord_unact.sigmoid()
            if self.as_two_stage else None)  # TODO: Is this right ?
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, memory, memory_mask, query, query_pos,
                        reference_points, spatial_shapes, level_start_index,
                        valid_ratios) -> Dict:  # TODO: typehint
        # TODO: Doc
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        head_inputs_dict = dict(
            hidden_states=inter_states,
            init_reference=reference_points,
            inter_references=inter_references)
        return head_inputs_dict

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_padding_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (batch_size, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (batch_size, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (batch_size, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (batch_size, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    @staticmethod
    def get_encoder_reference_points(  # TODO: should be moved to encoder
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[str, torch.device]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]  # TODO: should be done in encoder
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Get the valid radios of feature map of a level.

        Args:
            mask (Tensor): Binary mask of a feature map, has \
                shape (N, H, W)

        Returns:
            Tensor: valid ratio [r_h, r_w] of the feature map, \
                has shape [1, 2]
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000):
        """Get the position embedding of proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                [bs, num_query, 4]
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, or h-axis. Note the final returned
                dimension for each position is 4 times of this value.
                Default to 128.
            temperature (int, optional): The temperature used for scaling
                the position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
                [bs, num_query, num_pos_feats * 4]
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                DeformableDetrTransformerEncoderLayer(**self.layer_cfg[i]))
        self.embed_dims = self.layers[0].embed_dims

    # TODO: whether to write forward to display args parsing


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                DeformableDetrTransformerDecoderLayer(**self.layer_cfg[i]))
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             'DeformableDetrTransformerDecoder')

    def forward(
            self,
            query: Tensor,
            *args,  # TODO: flatten this
            reference_points: Tensor = None,
            valid_ratios: Tensor = None,
            reg_branches: Optional[nn.Module] = None,
            **kwargs) -> Tensor:
        """Forward function for `DeformableDetrTransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (batch_size, num_query, 4) when as_two_stage,  # TODO
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branches: (obj:`nn.ModuleList`, optional): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims]
                when return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

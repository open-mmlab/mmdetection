# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import namedtuple
from typing import List

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import caffe2_xavier_init
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.init import normal_

from mmdet.models.layers import (DeformableDetrTransformerEncoder,
                                 SinePositionalEncoding)


class MaskDINOEncoder(nn.Module):
    """This is the multi-scale encoder in detection models, also named as pixel
    decoder in segmentation models."""

    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],  # TODO: typeint ask Mask2former
        in_strides=[4, 8, 16, 32],  # TODO: typeint ask Mask2former
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm_cfg=dict(type='GN',
                      num_groups=32),  # TODO: typeint ask Mask2former
        # deformable transformer encoder args
        transformer_in_features: List[str] = ['res3', 'res4', 'res5'],
        common_stride: int = 4,
        num_feature_levels: int = 3,
        total_num_feature_levels: int = 4,
        feature_order: str = 'low2high',
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm_cfg (str or callable): normalization for all conv layers
            num_feature_levels: feature scales used
            total_num_feature_levels: total feature scales used (include the downsampled features)
            feature_order: 'low2high' or 'high2low', i.e., 'low2high' means low-resolution feature are put in the first.
        """
        super().__init__()

        assert len(in_channels) == len(in_strides) == 4
        DummyShapeSpec = namedtuple('DummyShapeSpec', ('channels', 'stride'))
        input_shape = {
            f'res{i + 2}':
            DummyShapeSpec(channels=in_channels[i], stride=in_strides[i])
            for i in range(len(in_channels))
        }
        warnings.warn(f'The input feature names are set '
                      f'{input_shape.keys()} with hardcode.')

        transformer_input_shape = {
            k: v
            for k, v in input_shape.items() if k in transformer_in_features
        }
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape
                            ]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        self.feature_order = feature_order

        if feature_order == 'low2high':
            transformer_input_shape = sorted(
                transformer_input_shape.items(), key=lambda x: -x[1].stride)
        else:
            transformer_input_shape = sorted(
                transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape
                                        ]  # starting from "res2" to "res5"
        transformer_in_channels = [
            v.channels for k, v in transformer_input_shape
        ]
        self.transformer_feature_strides = [
            v.stride for k, v in transformer_input_shape
        ]  # to decide extra FPN layers

        self.maskdino_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(
            max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    ))
            # input projectino for downsample
            in_channels = max(transformer_in_channels)
            for _ in range(
                    self.total_num_feature_levels -
                    self.transformer_num_feature_levels):  # exclude the res2
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            conv_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        nn.GroupNorm(32, conv_dim),
                    ))
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
            ])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = SinePositionalEncoding(N_steps, normalize=True)

        self.mask_dim = mask_dim
        self.mask_features = nn.Conv2d(
            conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        caffe2_xavier_init(self.mask_features)
        # weight_init.c2_xavier_fill(self.mask_features)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(
            int(np.log2(stride) - np.log2(self.common_stride)), 1)

        lateral_convs = []
        output_convs = []

        use_bias = False
        for idx, in_channels in enumerate(
                self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = ConvModule(
                in_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            caffe2_xavier_init(lateral_conv.conv, bias=0)
            caffe2_xavier_init(output_conv.conv, bias=0)

            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @autocast(enabled=False)
    def forward_features(self, features, masks):
        """
        :param features: multi-scale features from the backbone
        :param masks: image mask
        :return: enhanced multi-scale features and mask feature (1/4 resolution) for the decoder to produce binary mask
        """
        features = {f'res{i + 2}': feat for i, feat in enumerate(features)}

        # backbone features
        srcs = []
        pos = []
        # additional downsampled features
        srcsl = []
        posl = []
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest_feat = features[self.transformer_in_features[
                self.low_resolution_index]].float()
            _len_srcs = self.transformer_num_feature_levels
            for l in range(_len_srcs, self.total_num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](smallest_feat)
                else:
                    src = self.input_proj[l](srcsl[-1])
                srcsl.append(src)
                # TODO: Here generate a dummy mask
                mask = torch.zeros((src.size(0), src.size(2), src.size(3)),
                                   device=src.device,
                                   dtype=torch.bool)
                posl.append(self.pe_layer(mask))
        srcsl = srcsl[::-1]
        # Reverse feature maps
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float(
            )  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            # TODO: Here generate a dummy mask
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)),
                               device=x.device,
                               dtype=torch.bool)
            pos.append(self.pe_layer(mask))
        srcs.extend(
            srcsl) if self.feature_order == 'low2high' else srcsl.extend(srcs)
        pos.extend(posl) if self.feature_order == 'low2high' else posl.extend(
            pos)
        if self.feature_order != 'low2high':
            srcs = srcsl
            pos = posl
        y, spatial_shapes, level_start_index = self.transformer(
            srcs, masks, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[
                    i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(
                z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0],
                                       spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(
                out[self.high_resolution_index],
                size=cur_fpn.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    """MSDeformAttn Transformer encoder in deformable detr."""

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        act_cfg=dict(type='ReLU', inplace=True),
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.encoder = DeformableDetrTransformerEncoder(
            # TODO: consider using **encoder_cfg
            num_layers=num_encoder_layers,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    num_levels=num_feature_levels,
                    num_points=enc_n_points),
                ffn_cfg=dict(
                    embed_dims=d_model,
                    feedforward_channels=dim_feedforward,
                    num_fcs=2,
                    ffn_drop=dropout,
                    act_cfg=act_cfg),
                norm_cfg=dict(type='LN')))

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):
        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((x.size(0), x.size(2), x.size(3)),
                            device=x.device,
                            dtype=torch.bool) for x in srcs
            ]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            query=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            query_pos=lvl_pos_embed_flatten,
            key_padding_mask=mask_flatten)

        return memory, spatial_shapes, level_start_index

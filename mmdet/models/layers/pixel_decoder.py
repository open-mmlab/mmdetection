# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .positional_encoding import SinePositionalEncoding
from .transformer import DetrTransformerEncoder


@MODELS.register_module()
class PixelDecoder(BaseModule):
    """Pixel decoder with a structure like fpn.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        for i in range(0, self.num_inputs - 1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.last_feat_conv = ConvModule(
            in_channels[-1],
            feat_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.last_feat_conv, bias=0)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information.
                Pass in for creating more accurate padding mask. Not
                used here.

        Returns:
            tuple[Tensor, Tensor]: a tuple containing the following:

                - mask_feature (Tensor): Shape (batch_size, c, h, w).
                - memory (Tensor): Output of last stage of backbone.\
                        Shape (batch_size, c, h, w).
        """
        y = self.last_feat_conv(feats[-1])
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + \
                F.interpolate(y, size=cur_feat.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)

        mask_feature = self.mask_feature(y)
        memory = feats[-1]
        return mask_feature, memory


@MODELS.register_module()
class TransformerEncoderPixelDecoder(PixelDecoder):
    """Pixel decoder with transormer encoder inside.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer encoder.
            Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 feat_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 encoder: ConfigType = None,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.last_feat_conv = None

        self.encoder = DetrTransformerEncoder(**encoder)
        self.encoder_embed_dims = self.encoder.embed_dims
        assert self.encoder_embed_dims == feat_channels, 'embed_dims({}) of ' \
            'tranformer encoder must equal to feat_channels({})'.format(
                feat_channels, self.encoder_embed_dims)
        self.positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.encoder_in_proj = Conv2d(
            in_channels[-1], feat_channels, kernel_size=1)
        self.encoder_out_proj = ConvModule(
            feat_channels,
            feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.encoder_in_proj, bias=0)
        caffe2_xavier_init(self.encoder_out_proj.conv, bias=0)

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats: List[Tensor],
                batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information. Pass in
                for creating more accurate padding mask.

        Returns:
            tuple: a tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - memory (Tensor): shape (batch_size, c, h, w).
        """
        feat_last = feats[-1]
        bs, c, h, w = feat_last.shape
        input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape']
        padding_mask = feat_last.new_ones((bs, input_img_h, input_img_w),
                                          dtype=torch.float32)
        for i in range(bs):
            img_h, img_w = batch_img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feat_last.shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)

        pos_embed = self.positional_encoding(padding_mask)
        feat_last = self.encoder_in_proj(feat_last)
        # (batch_size, c, h, w) -> (batch_size, num_queries, c)
        feat_last = feat_last.flatten(2).permute(0, 2, 1)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        # (batch_size, h, w) -> (batch_size, h*w)
        padding_mask = padding_mask.flatten(1)
        memory = self.encoder(
            query=feat_last,
            query_pos=pos_embed,
            key_padding_mask=padding_mask)
        # (batch_size, num_queries, c) -> (batch_size, c, h, w)
        memory = memory.permute(0, 2, 1).view(bs, self.encoder_embed_dims, h,
                                              w)
        y = self.encoder_out_proj(memory)
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + \
                F.interpolate(y, size=cur_feat.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)

        mask_feature = self.mask_feature(y)
        return mask_feature, memory

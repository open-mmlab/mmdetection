# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor

from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType


class RepVGGBlock(BaseModule):
    """RepVGG block is modifided to skip branch_norm.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): The stride of the block. Defaults to 1.
        padding (int): The padding of the block. Defaults to 1.
        dilation (int): The dilation of the block. Defaults to 1.
        groups (int): The groups of the block. Defaults to 1.
        padding_mode (str): The padding mode of the block. Defaults to 'zeros'.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to None.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict): The config dict for activation layers.
            Defaults to dict(type='ReLU').
        without_branch_norm (bool): Whether to skip branch_norm.
            Defaults to True.
        init_cfg (dict): The config dict for initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU'),
                 without_branch_norm: bool = True,
                 init_cfg: OptConfigType = None):
        super(RepVGGBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # judge if input shape and output shape are the same.
        # If true, add a normalized identity shortcut.
        if out_channels == in_channels and stride == 1 and \
                padding == dilation and not without_branch_norm:
            self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.branch_norm = None

        self.branch_3x3 = self.create_conv_bn(
            kernel_size=3,
            dilation=dilation,
            padding=padding,
        )
        self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self,
                       kernel_size: int,
                       dilation: int = 1,
                       padding: int = 0) -> nn.Sequential:
        """Create a conv_bn layer.

        Args:
            kernel_size (int): The kernel size of the conv layer.
            dilation (int, optional): The dilation of the conv layer.
                Defaults to 1.
            padding (int, optional): The padding of the conv layer.
                Defaults to 0.

        Returns:
            nn.Sequential: The created conv_bn layer.
        """
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x: Tensor) -> Tensor:
        """1x1 conv + 3x3 conv + identity shortcut.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out

        out = self.act(out)

        return out


class CSPRepLayer(BaseModule):
    """CSPRepLayer.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        num_blocks (int): The number of blocks in the layer. Defaults to 3.
        expansion (float): The expansion of the block. Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 3,
                 expansion: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True)):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottlenecks = nn.Sequential(*[
            RepVGGBlock(hidden_channels, hidden_channels, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@MODELS.register_module()
class HybridEncoder(BaseModule):
    """HybridEncoder.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the layer.
        projector (:obj:`ConfigDict` or dict, optional): The config dict for
            the projector. Defaults to None.
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [512, 1024, 2048].
        feat_strides (List[int], optional): The strides of the feature
            maps. Defaults to [8, 16, 32].
        hidden_dim (int, optional): The hidden dimension of the MLP.
            Defaults to 256.
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        pe_temperature (int, optional): The temperature of the positional
            encoding. Defaults to 10000.
        expansion (float, optional): The expansion of the CSPRepLayer.
            Defaults to 1.0.
        depth_mult (float, optional): The depth multiplier of the CSPRepLayer.
            Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        eval_size (int, optional): The size of the test image.
            Defaults to None.
    """

    def __init__(self,
                 layer_cfg: ConfigType,
                 projector: OptConfigType = None,
                 num_encoder_layers: int = 1,
                 in_channels: List[int] = [512, 1024, 2048],
                 feat_strides: List[int] = [8, 16, 32],
                 hidden_dim: int = 256,
                 use_encoder_idx: List[int] = [2],
                 pe_temperature: int = 10000,
                 expansion: float = 1.0,
                 depth_mult: float = 1.0,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                ConvModule(
                    in_channel,
                    hidden_dim,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

        # encoder transformer
        self.encoder = ModuleList([
            DetrTransformerEncoder(num_encoder_layers, layer_cfg)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        lateral_convs = list()
        fpn_blocks = list()
        for idx in range(len(in_channels) - 1, 0, -1):
            lateral_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    1,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.lateral_convs = ModuleList(lateral_convs)
        self.fpn_blocks = ModuleList(fpn_blocks)

        # bottom-up pan
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(in_channels) - 1):
            downsample_convs.append(
                ConvModule(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act_cfg=act_cfg,
                    expansion=expansion))
        self.downsample_convs = ModuleList(downsample_convs)
        self.pan_blocks = ModuleList(pan_blocks)

        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        proj_feats = [
            self.input_proj[i](inputs[i]) for i in range(len(inputs))
        ]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(
                    0, 2, 1).contiguous()
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](
                    src_flatten,
                    query_pos=pos_embed.to(src_flatten.device),
                    key_padding_mask=None)
                proj_feats[enc_ind] = memory.permute(
                    0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(
                feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.cat([downsample_feat, feat_high], axis=1))
            outs.append(out)

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)

    @staticmethod
    def build_2d_sincos_position_embedding(w: int,
                                           h: int,
                                           embed_dim=256,
                                           temperature=10000.) -> Tensor:
        """Build 2D sin-cos position embedding.

        Args:
            w (int): The width of the feature map.
            h (int): The height of the feature map.
            embed_dim (int): The dimension of the embedding.
                Defaults to 256.
            temperature (float): The temperature of the position embedding.
                Defaults to 10000.

        Returns:
            Tensor: The position embedding.
        """

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            ('Embed dimension must be divisible by 4 for '
             '2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                         axis=1)[None, :, :]

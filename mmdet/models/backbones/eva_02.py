# Copyright (c) OpenMMLab. All rights reserved.
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .eva_02_utils import (PatchEmbed, VisionRotaryEmbeddingFast, get_abs_pos,
                           get_norm, window_partition, window_unpartition)

try:
    import xformers.ops as xops
    HAS_XFORMER = True
except ImportError:
    HAS_XFORMER = False
    pass

logger = logging.getLogger(__name__)

__all__ = ['EVA02_ViT', 'SimpleFeaturePyramid', 'get_vit_lr_decay_rate']


@dataclass
class ShapeSpec:
    """A simple structure that contains basic shape specification about a
    tensor.

    It is often used as the auxiliary inputs/outputs of models, to complement
    the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class SwiGLU(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.SiLU,
                 drop=0.,
                 norm_layer=nn.LayerNorm,
                 subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_head_dim=None,
        rope=None,
        xattn=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        self.xattn = xattn
        self.proj = nn.Linear(all_head_dim, dim)

        if not HAS_XFORMER:
            self.xattn = False

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads,
                      -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # rope
        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1,
                          3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = x.view(B, H, W, C)

        return x


class ResBottleneckBlock(nn.Module):
    """The standard bottleneck residual block without the last activation
    layer.

    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            bottleneck_channels,
            norm=dict(type='LN'),
            act_layer=dict(type='GELU'),
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.layer1 = ConvModule(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=norm,
            act_cfg=act_layer)

        self.layer2 = ConvModule(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm,
            act_cfg=act_layer)

        self.layer2 = ConvModule(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=norm,
            act_cfg=None)
        '''
        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()
        '''

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual
    propagation blocks."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4 * 2 / 3,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_size=0,
        use_residual_block=False,
        rope=None,
        xattn=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query,
                key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to
                the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative
                positional parameters.
            window_size (int): Window size for window attention blocks. If it
            equals 0, then not use window attention.
            use_residual_block (bool): If True, use a residual block
                after the MLP block.
            input_size (int or None): Input resolution for calculating the
            relative positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope=rope,
            xattn=xattn,
        )

        from timm.models.layers import DropPath

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            subln=True,
            norm_layer=norm_layer,
        )

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm='LN',
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


@MODELS.register_module()
class EVA02_ViT(BaseModule):
    """This module implements Vision Transformer (ViT) backbone in
    :paper:`vitdet`.

    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4 * 2 / 3,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 use_abs_pos=True,
                 use_rel_pos=False,
                 rope=True,
                 pt_hw_seq_len=16,
                 intp_freq=True,
                 window_size=0,
                 window_block_indexes=(),
                 residual_block_indexes=(),
                 use_act_checkpoint=False,
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 out_feature='last_feat',
                 xattn=True,
                 init_cfg=None):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key,value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to
                the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative
                positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks
                using window attention.
            residual_block_indexes (list): Indexes for blocks using
                conv propagation.
            use_act_checkpoint (bool): If True, use
                activation checkpointing.
            pretrain_img_size (int): input image size
                for pretraining models.
            pretrain_use_cls_token (bool): If True,
                pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size)
            num_positions = (num_patches +
                             1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        half_head_dim = embed_dim // num_heads // 2
        hw_seq_len = img_size // patch_size

        print(f'dim: {half_head_dim}, \
                pt_seq_len: {pt_hw_seq_len}, windowsize: {window_size}')
        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )

        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=hw_seq_len if intp_freq else None,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                rope=self.rope_win
                if i in window_block_indexes else self.rope_glb,
                xattn=xattn)
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))
        for blk in self.blocks:
            x = blk(x)
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@MODELS.register_module()
class SimpleFeaturePyramid(BaseModule):
    """This module implements SimpleFeaturePyramid in :paper:`vitdet`.

    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm='LN',
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps
                coming from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to
                upsample or downsample the input features
                for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list.
                    The top_block
                further downsamples the feature map. It must have
                an attribute
                "num_levels", meaning the number of extra pyramid
                levels added by this block, and "in_feature", which is a
                string representing its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to
                specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()

        strides = [
            int(input_shapes[in_feature].stride / scale)
            for scale in scale_factors
        ]

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ''
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(
                    f'scale_factor={scale} is not supported yet.')

            layers.extend([
                Conv2d(
                    out_dim,
                    out_channels,
                    kernel_size=1,
                    bias=use_bias,
                ),
                get_norm(norm, out_channels),
                Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
                get_norm(norm, out_channels)
            ])
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f'simfp_{stage}', layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            'p{}'.format(int(math.log2(s))): s
            for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides['p{}'.format(s + 1)] = 2**(s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channels
            for k in self._out_features
        }
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            'size_divisiblity': self._size_divisibility,
            'square_size': self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of
            ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map
                tensor in high to low resolution order.
                Returned feature names follow the FPN
                convention: "p<stage>", where stage has
                stride = 2 ** stage e.g., ["p2", "p3", ..., "p6"].
        """

        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[
                    self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(
                    self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith('backbone'):
        if '.pos_embed' in name or '.patch_embed' in name:
            layer_id = 0
        elif '.blocks.' in name and '.residual.' not in name:
            layer_id = int(name[name.find('.blocks.'):].split('.')[2]) + 1

    return lr_decay_rate**(num_layers + 1 - layer_id)

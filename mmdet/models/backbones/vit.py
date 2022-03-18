from mmcls.models import VisionTransformer as _VisionTransformer
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer as _TransformerEncoderLayer
from torch import nn
from mmcv.cnn.bricks.transformer import FFN
import torch
from ..builder import BACKBONES
from mmcv.cnn import build_norm_layer
from mmcls.models.utils import MultiheadAttention
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.runner.base_module import ModuleList
import math
import torch.nn.functional as F
import numpy as np


class TransformerEncoderLayer(_TransformerEncoderLayer):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 window_size=14,
                 use_window=False,
                 pad_mode='constant',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        if use_window:
            self.attn = WindowMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias,
                window_size=window_size,
                pad_mode=pad_mode)
        else:
            self.attn = MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)


class WindowMultiheadAttention(MultiheadAttention):
    """Window Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 pad_mode='constant',
                 window_size=14,
                 init_cfg=None):
        super(WindowMultiheadAttention, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=input_dims,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            dropout_layer=dropout_layer,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            v_shortcut=v_shortcut,
            init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)
        self.pad_mode = pad_mode
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(N**0.5), int(N**0.5)
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x).transpose(1, 2).reshape(B, 3 * C, H, W)
        qkv = F.pad(qkv, [0, H_ - H, 0, W_ - W], mode=self.pad_mode)

        qkv = F.unfold(
            qkv,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size))
        B, C_kw_kw, L = qkv.shape
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # B, L, N_, 3C
        qkv = qkv.reshape(B, L, N_, 3,
                          self.num_heads, C // self.num_heads).permute(
                              3, 0, 1, 4, 2, 5)  # 3, B, L, num_heads, N_, C
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)
        x = F.fold(
            x,
            output_size=(H_, W_),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size))
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


@BACKBONES.register_module()
class VisionTransformer(_VisionTransformer):

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 num_groups=8,
                 out_indices=(2, 5, 8, 11),
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=False,
                 output_cls_token=False,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            init_cfg=init_cfg)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.arch_settings['num_layers'])

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=self.arch_settings.get('qkv_bias', True),
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.res_modify_block_0 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2),
            nn.GroupNorm(num_groups, self.embed_dims), nn.GELU(),
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2))
        self.res_modify_block_1 = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2))
        self.res_modify_block_2 = nn.Sequential(nn.Identity())
        self.res_modify_block_3 = nn.Sequential(nn.MaxPool2d(2, 2))

        _, self.norm0 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        _, self.norm1 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        _, self.norm2 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)
        _, self.norm3 = build_norm_layer(norm_cfg, self.embed_dims, postfix=1)

    def forward(self, x):
        x = self.patch_embed(x)
        B, _, C = x.shape
        patch_resolution = self.patch_embed.patches_resolution

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = x[:, 1:]
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        # Normalization and Reshape
        out0 = self.norm0(outs[0]).reshape(B, *patch_resolution,
                                           C).permute(0, 3, 1, 2)
        out1 = self.norm1(outs[1]).reshape(B, *patch_resolution,
                                           C).permute(0, 3, 1, 2)
        out2 = self.norm2(outs[2]).reshape(B, *patch_resolution,
                                           C).permute(0, 3, 1, 2)
        out3 = self.norm3(outs[3]).reshape(B, *patch_resolution,
                                           C).permute(0, 3, 1, 2)

        # Convert to multi-scale
        out0 = self.res_modify_block_0(out0)
        out1 = self.res_modify_block_1(out1)
        out2 = self.res_modify_block_2(out2)
        out3 = self.res_modify_block_3(out3)

        return tuple([out0, out1, out2, out3])
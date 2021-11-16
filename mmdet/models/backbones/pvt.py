import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)
from torch.nn.modules.utils import _pair as to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert


class MixFFN(BaseModule):
    """An implementation of MixFFN of PVT.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Depth-wise Conv to encode positional information.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, activate, drop, fc2, drop]
        if use_conv:
            layers.insert(1, dw_conv)
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class SpatialReductionAttention(MultiheadAttention):
    """An implementation of Spatial Reduction Attention of PVT.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmdet import mmcv_version, digit_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'SpatialReductionAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class PVTEncoderLayer(BaseModule):
    """Implements one encoder layer in PVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_conv_ffn=False,
                 init_cfg=None):
        super(PVTEncoderLayer, self).__init__(init_cfg=init_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = SpatialReductionAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)

        return x


class AbsolutePositionEmbedding(BaseModule):
    """An implementation of the absolute position embedding in PVT.

    Args:
        pos_shape (int): The shape of the absolute position embedding.
        pos_dim (int): The dimension of the absolute position embedding.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.0.
    """

    def __init__(self, pos_shape, pos_dim, drop_rate=0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pos_shape, int):
            pos_shape = to_2tuple(pos_shape)
        elif isinstance(pos_shape, tuple):
            if len(pos_shape) == 1:
                pos_shape = to_2tuple(pos_shape[0])
            assert len(pos_shape) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pos_shape)}'
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
        """Resize pos_embed weights.

        Resize pos_embed using bilinear interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bilinear'``.

        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C].
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight,
                                         2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)


@BACKBONES.register_module()
class PyramidVisionTransformer(BaseModule):
    """Pyramid Vision Transformer (PVT)

    Implementation of `Pyramid Vision Transformer: A Versatile Backbone for
    Dense Prediction without Convolutions
    <https://arxiv.org/pdf/2102.12122.pdf>`_.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 paddings=[0, 0, 0, 0],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=True,
                 norm_after_stage=False,
                 use_conv_ffn=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 convert_weights=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims

        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        self.pretrained = pretrained

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True,
                norm_cfg=norm_cfg)

            layers = ModuleList()
            if use_abs_pos_embed:
                pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
                pos_embed = AbsolutePositionEmbedding(
                    pos_shape=pos_shape,
                    pos_dim=embed_dims_i,
                    drop_rate=drop_rate)
                layers.append(pos_embed)
            layers.extend([
                PVTEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratios[i] * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i],
                    use_conv_ffn=use_conv_ffn) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            if norm_after_stage:
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            else:
                norm = nn.Identity()
            self.layers.append(ModuleList([patch_embed, layers, norm]))
            cur += num_layer

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, AbsolutePositionEmbedding):
                    m.init_weights()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if self.convert_weights:
                # Because pvt backbones are not supported by mmcls,
                # so we need to convert pre-trained weights to match this
                # implementation.
                state_dict = pvt_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)

            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


@BACKBONES.register_module()
class PyramidVisionTransformerV2(PyramidVisionTransformer):
    """Implementation of `PVTv2: Improved Baselines with Pyramid Vision
    Transformer <https://arxiv.org/pdf/2106.13797.pdf>`_."""

    def __init__(self, **kwargs):
        super(PyramidVisionTransformerV2, self).__init__(
            patch_sizes=[7, 3, 3, 3],
            paddings=[3, 1, 1, 1],
            use_abs_pos_embed=False,
            norm_after_stage=True,
            use_conv_ffn=True,
            **kwargs)

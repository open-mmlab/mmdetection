from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from torch.nn import functional as F

from mmdet.registry import MODELS
from .transformer_blocks import (Conv2d, PositionEmbeddingSine,
                                 TransformerEncoder, TransformerEncoderLayer,
                                 get_norm)


class TransformerEncoderOnly(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class BasePixelDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        conv_dim: int,
        mask_dim: int,
        mask_on: bool,
        norm: Optional[Union[str, Callable]] = None,
    ):
        super().__init__()

        lateral_convs = []
        output_convs = []

        use_bias = norm == ''
        for idx, in_channel in enumerate(in_channels):
            if idx == len(in_channels) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channel,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module('layer_{}'.format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channel,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm)
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # weight_init.c2_xavier_fill(lateral_conv)
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
                self.add_module('layer_{}'.format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_on = mask_on
        if self.mask_on:
            self.mask_dim = mask_dim
            self.mask_features = Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            # weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales


@MODELS.register_module(force=True)
class TransformerEncoderPixelDecoder(BasePixelDecoder):

    def __init__(
        self,
        in_channels,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_enc_layers: int = 6,
        transformer_pre_norm: bool = False,
        conv_dim: int = 512,
        mask_dim: int = 512,
        mask_on: bool = True,
        norm: Optional[Union[str, Callable]] = 'GN',
    ):

        super().__init__(
            in_channels,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm=norm,
            mask_on=mask_on)

        self.in_features = ['res2', 'res3', 'res4', 'res5']
        feature_channels = in_channels

        in_channels = feature_channels[len(in_channels) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        # weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ''
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        # weight_init.c2_xavier_fill(output_conv)
        delattr(self, 'layer_{}'.format(len(self.in_features)))
        self.add_module('layer_{}'.format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    # 使用 transformer 图片特征的多尺度特征融合,类似于 transformer 版本的 FPN
    def forward(self, features):
        multi_scale_features = []
        num_cur_levels = 0

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(
                    y, size=cur_fpn.shape[-2:], mode='nearest')
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1

        mask_features = self.mask_features(y) if self.mask_on else None
        return mask_features, multi_scale_features

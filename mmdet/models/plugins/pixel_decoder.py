import torch
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS, Conv2d, ConvModule, kaiming_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule, ModuleList


@PLUGIN_LAYERS.register_module()
class PixelDecoder(BaseModule):
    """Pixel decoder with a structure like fpn.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (obj:`mmcv.ConfigDict`|dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (obj:`mmcv.ConfigDict`|dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (obj:`mmcv.ConfigDict`|dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (obj:`mmcv.ConfigDict`|dict):  Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        for i in range(0, self.num_inputs - 1):
            l_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg)
            o_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            self.output_convs.append(o_conv)

        self.last_feat_output_conv = ConvModule(
            in_channels[-1],
            feat_channels,
            kernel_size=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg)
        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            kaiming_init(self.lateral_convs[i].conv, a=1)
            kaiming_init(self.output_convs[i].conv, a=1)

        kaiming_init(self.mask_feature, a=1)
        kaiming_init(self.last_feat_output_conv, a=1)

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of [bs, c, h, w].

        Returns:
            tuple: a tuple containing the following:

                - mask_feature (Tensor): Shape [bs, c, h, w].
                - memory (Tensor): Output of last stage of backbone.
                    Shape [bs, c, h, w].
        """
        y = self.last_feat_output_conv(feats[-1])
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_fpn = self.lateral_convs[i](x)
            y = cur_fpn + \
                F.interpolate(y, size=cur_fpn.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)

        mask_feature = self.mask_feature(y)
        memory = feats[-1]
        return mask_feature, memory


@PLUGIN_LAYERS.register_module()
class TransformerEncoderPixelDecoder(PixelDecoder):
    """Pixel decoder with transormer encoder inside.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (obj:`mmcv.ConfigDict`|dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (obj:`mmcv.ConfigDict`|dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (obj:`mmcv.ConfigDict`|dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (obj:`mmcv.ConfigDict`|dict):  Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 encoder=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None):
        super(TransformerEncoderPixelDecoder, self).__init__(
            in_channels,
            feat_channels,
            out_channels,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        self.last_feat_output_conv = None

        self.encoder = build_transformer_layer_sequence(encoder)
        self.encoder_embed_dims = self.encoder.embed_dims
        assert self.encoder_embed_dims == feat_channels, 'embed_dims({}) of ' \
            'tranformer encoder must equal to feat_channels({})'.format(
                feat_channels, self.encoder_embed_dims)
        self.pe = build_positional_encoding(positional_encoding)
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

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            kaiming_init(self.lateral_convs[i].conv, a=1)
            kaiming_init(self.output_convs[i].conv, a=1)

        kaiming_init(self.mask_feature, a=1)
        kaiming_init(self.encoder_in_proj, a=1)
        kaiming_init(self.encoder_out_proj.conv, a=1)

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of [bs, c, h, w].

        Returns:
            tuple: a tuple containing the following:

                - mask_feature (Tensor): shape [bs, c, h, w].
                - memory (Tensor): shape [bs, c, h, w].
        """
        feat_last = feats[-1]
        bs, c, h, w = feat_last.shape
        padding_mask = feat_last.new_zeros((bs, h, w), dtype=torch.bool)
        pos_embed = self.pe(padding_mask)
        feat_last = self.encoder_in_proj(feat_last)
        # [bs, c, h, w] -> [nq, bs, dim]
        feat_last = feat_last.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        padding_mask = padding_mask.flatten(1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=feat_last,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=padding_mask)
        # [nq, bs, em] -> [bs, c, h, w]
        memory = memory.permute(1, 2, 0).view(bs, self.encoder_embed_dims, h,
                                              w)
        memory = self.encoder_out_proj(memory)
        y = memory
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_fpn = self.lateral_convs[i](x)
            y = cur_fpn + \
                F.interpolate(y, size=cur_fpn.shape[-2:], mode='nearest')
            y = self.output_convs[i](y)

        mask_feature = self.mask_feature(y)
        return mask_feature, memory

import warnings

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, xavier_init

from mmdet.ops import DeformConv, ModulatedDeformConv
from .conv_ws import ConvWS2d
from .norm import build_norm_layer

conv_cfg = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


class DCNModule(nn.Module):
    """A conv block that contains deformable conv with or
        without norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        dcn_kernel (int or tuple[int]): kernel size for deformable conv.
        offset_kernel (int or tuple[int]): kernel size for offset conv.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        deformable_groups (int): number of groups for deformable conv.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        modulated (bool): whether to use modulated DCN.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    _version = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dcn_kernel=3,
                 offset_kernel=3,
                 groups=1,
                 deformable_groups=1,
                 dcn_dilation=1,
                 offset_dilation=1,
                 compressed_channels=-1,
                 modulated=False,
                 bias='auto',
                 norm_cfg=None,
                 activation=None,
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(DCNModule, self).__init__()
        self.in_channels = in_channels
        self.dcn_kernel = dcn_kernel
        self.offset_kernel = offset_kernel
        self.deformable_groups = deformable_groups
        self.offset_dilation = offset_dilation
        self.modulated = modulated
        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        self.activation = activation
        self.order = order
        self.groups = groups
        self.dcn_dilation = dcn_dilation
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if (self.with_norm or not self.modulated) else True
        self.with_bias = bias
        self.compressed_channels = compressed_channels
        self.with_compress = True if compressed_channels > 0 else False
        if self.with_compress:
            self.channel_compressor = nn.Conv2d(in_channels,
                                                self.compressed_channels, 1)
            offset_in_channels = self.compressed_channels
        else:
            offset_in_channels = in_channels
        if self.modulated:
            self.offset_conv = nn.Conv2d(
                offset_in_channels,
                self.deformable_groups * 3 * self.dcn_kernel * self.dcn_kernel,
                self.offset_kernel,
                stride=stride,
                padding=int(
                    (self.offset_kernel - 1) * self.offset_dilation / 2),
                dilation=self.offset_dilation)
            self.dcn_conv = ModulatedDeformConv(
                in_channels,
                out_channels,
                kernel_size=self.dcn_kernel,
                stride=stride,
                padding=self.dcn_dilation,
                dilation=self.dcn_dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                bias=self.with_bias)
        else:
            self.offset_conv = nn.Conv2d(
                offset_in_channels,
                self.deformable_groups * 2 * self.dcn_kernel * self.dcn_kernel,
                self.offset_kernel,
                stride=stride,
                padding=int(
                    (self.offset_kernel - 1) * self.offset_dilation / 2),
                dilation=self.offset_dilation)
            self.dcn_conv = DeformConv(
                in_channels,
                out_channels,
                kernel_size=self.dcn_kernel,
                stride=stride,
                padding=self.dcn_dilation,
                dilation=self.dcn_dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                bias=False)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        # TODO: check the init
        # Detectron2 use kaiming_init, in CAFA we use xavier
#         xavier_init(self.dcn_conv, distribution='uniform')
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
#         if self.with_compress:
#             xavier_init(self.channel_compressor, distribution='uniform')

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_compress:
                    offset_input = self.channel_compressor(x)
                else:
                    offset_input = x
                if self.modulated:
                    out = self.offset_conv(offset_input)
                    o1, o2, mask = torch.chunk(out, 3, dim=1)
                    offset = torch.cat((o1, o2), dim=1)
                    mask = torch.sigmoid(mask)
                    x = self.dcn_conv(x, offset, mask)
                else:
                    offset = self.offset_conv(offset_input)
                    x = self.dcn_conv(x, offset)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, the DCNModule load previous benchmark models.
            if (prefix + "offset_conv.weight" not in state_dict 
                and prefix[:-1] + "_offset.weight" in state_dict):
                state_dict[prefix + "offset_conv.weight"] = state_dict[
                    prefix[:-1] + "_offset.weight"]
            if (prefix + "offset_conv.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict):
                state_dict[prefix + "offset_conv.bias"] = state_dict[
                    prefix[:-1] + "_offset.bias"]
            # the conv2.weight is in both pretrain & current model
            if (prefix + "dcn_conv.weight" not in state_dict
                and prefix + "weight" in state_dict):
                state_dict[prefix + "dcn_conv.weight"] = state_dict[
                    prefix + "weight"]

        if version is not None and version > 1:
            import logging
            logger = logging.getLogger()
            logger.info("DCNModule {} is upgraded to version 2.".format(
                prefix.rstrip(".")))

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.ops.carafe import CARAFEPack

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class FPN_CARAFE(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_ups=0,
                 norm_cfg=None,
                 activation=None,
                 upsample='nearest',
                 upsample_kernel=-1,
                 order=('conv', 'norm', 'act'),
                 carafe_cfg=dict(
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1,
                     benchmark='auto')):
        super(FPN_CARAFE, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.norm_cfg = norm_cfg
        self.with_bias = norm_cfg is None
        self.upsample = upsample
        self.upsample_kernel = upsample_kernel
        self.with_carafe = carafe_cfg is not None
        self.carafe_cfg = carafe_cfg
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert self.upsample_kernel > 0

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_ups = add_extra_ups

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()
        self.up_modules = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                activation=activation,
                inplace=False,
                order=self.order)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                activation=activation,
                inplace=False,
                order=self.order)
            if i != self.backbone_end_level - 1:
                if self.upsample_kernel > 0:
                    if self.upsample == 'deconv':
                        upsample_conv = nn.ConvTranspose2d(
                            out_channels,
                            out_channels,
                            self.upsample_kernel,
                            stride=2,
                            padding=int((self.upsample_kernel - 1) / 2),
                            output_padding=int((self.upsample_kernel - 1) / 2))
                    elif self.upsample == 'pixel_shuffle':
                        upsample_conv = nn.Conv2d(
                            out_channels,
                            out_channels * 4,
                            self.upsample_kernel,
                            padding=int((self.upsample_kernel - 1) / 2))
                    else:
                        upsample_conv = nn.Conv2d(
                            out_channels,
                            out_channels,
                            self.upsample_kernel,
                            padding=int((self.upsample_kernel - 1) / 2))
                    self.upsample_convs.append(upsample_conv)
            if self.with_carafe and i != self.backbone_end_level - 1:
                self._add_up_modules(out_channels)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        extra_levels = extra_out_levels + self.add_extra_ups
        if extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False,
                    order=self.order)
                if self.upsample_kernel > 0:
                    if self.upsample == 'deconv':
                        upsample_conv = nn.ConvTranspose2d(
                            out_channels,
                            out_channels,
                            self.upsample_kernel,
                            stride=2,
                            padding=int((self.upsample_kernel - 1) / 2),
                            output_padding=int((self.upsample_kernel - 1) / 2))
                    elif self.upsample == 'pixel_shuffle':
                        upsample_conv = nn.Conv2d(
                            out_channels,
                            out_channels * 4,
                            self.upsample_kernel,
                            padding=int((self.upsample_kernel - 1) / 2))
                    else:
                        upsample_conv = nn.Conv2d(
                            out_channels,
                            out_channels,
                            self.upsample_kernel,
                            padding=int((self.upsample_kernel - 1) / 2))
                    self.upsample_convs.append(upsample_conv)
                if i < extra_out_levels:
                    extra_fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.with_bias,
                        activation=activation,
                        inplace=False,
                        order=self.order)
                    self.fpn_convs.append(extra_fpn_conv)
                if self.with_carafe:
                    self._add_up_modules(out_channels)
                self.lateral_convs.append(extra_l_conv)

    def _add_up_modules(self, in_channels):
        _up_modules = nn.ModuleDict()
        carafe = CARAFEPack(in_channels, 2, **self.carafe_cfg)
        _up_modules.update({'carafe': carafe})
        self.up_modules.append(_up_modules)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def up_forward(self, up_feat, lvl):
        _up_modules = self.up_modules[lvl]
        carafe = _up_modules['carafe']
        up_feat = carafe(up_feat)
        return up_feat

    def resize_as(self, a, b):
        # resize a as b
        if a.size(2) == b.size(2) and a.size(3) == b.size(3):
            return a
        else:
            assert (abs(a.size(2) - b.size(2)) <= 1
                    and abs(a.size(3) - b.size(3)) <= 1)
            return a[:, :, :b.size(2), :b.size(3)]

    def tensor_add(self, a, b):
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.resize_as(b, a)
        return c

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            if self.upsample is not None:
                if (self.upsample != 'deconv'
                        and self.upsample != 'pixel_shuffle'):
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_feat = F.interpolate(
                        laterals[i],
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                    if self.upsample_kernel > 0:
                        upsample_feat = self.upsample_convs[i - 1](
                            upsample_feat)
                elif self.upsample == 'deconv':
                    upsample_feat = self.upsample_convs[i - 1](
                        laterals[i], output_size=laterals[i - 1].size())
                elif self.upsample == 'pixel_shuffle':
                    upsample_feat = self.upsample_convs[i - 1](laterals[i])
                    upsample_feat = F.pixel_shuffle(upsample_feat, 2)
                else:
                    AssertionError('upsample method not impl')
            else:
                upsample_feat = laterals[i]
            if self.with_carafe:
                upsample_feat = self.up_forward(upsample_feat, i - 1)
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)

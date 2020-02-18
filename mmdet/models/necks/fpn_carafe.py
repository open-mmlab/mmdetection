import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops.carafe import CARAFEPack
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPN_CARAFE(nn.Module):
    """FPN_CARAFE is a more flexible implementation of FPN.
    It allows more choice for upsample methods during the top-down pathway.

    It can reproduce the preformance of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        in_channels (list): Number of channels for each input feature map.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        start_level (int): start level of feature pyramids.
        end_level (int): end level of feature pyramids.
            (-1 indicates the last level).
        norm_cfg (dict): dictionary to construct and config norm layer.
        activate (str): type of activation function in ConvModule
            (None indicates w/o activation).
        order (dict): order of components in ConvModule.
        upsample (str): type of upsample layer.
        upsample_cfg (dict): dictionary to construct and config upsample layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 activation=None,
                 order=('conv', 'norm', 'act'),
                 upsample='nearest',
                 upsample_cfg=dict(
                     up_kernel=5,
                     up_group=1,
                     encoder_kernel=3,
                     encoder_dilation=1)):
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
        self.upsample_cfg = upsample_cfg
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]

        assert self.upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if self.upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                self.upsample_cfg,
                'upsample_kernel') and self.upsample_cfg.upsample_kernel > 0
            self.upsample_kernel = self.upsample_cfg.upsample_kernel

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

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()

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
                if self.upsample == 'deconv':
                    upsample_module = nn.ConvTranspose2d(
                        out_channels,
                        out_channels,
                        self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_module = nn.Conv2d(
                        out_channels,
                        out_channels * 4,
                        self.upsample_kernel,
                        padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'carafe':
                    upsample_module = CARAFEPack(out_channels, 2,
                                                 **self.upsample_cfg)
                self.upsample_modules.append(upsample_module)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
            num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
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
                if self.upsample == 'deconv':
                    upsample_module = nn.ConvTranspose2d(
                        out_channels,
                        out_channels,
                        self.upsample_kernel,
                        stride=2,
                        padding=(self.upsample_kernel - 1) // 2,
                        output_padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'pixel_shuffle':
                    upsample_module = nn.Conv2d(
                        out_channels,
                        out_channels * 4,
                        self.upsample_kernel,
                        padding=(self.upsample_kernel - 1) // 2)
                elif self.upsample == 'carafe':
                    upsample_module = CARAFEPack(out_channels, 2,
                                                 **self.upsample_cfg)
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
                self.upsample_modules.append(upsample_module)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def slice_as(self, src, dst):
        # slice src as dst
        # src should have the same or larger size than dst
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
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
                if (self.upsample == 'nearest' or self.upsample == 'bilinear'):
                    align_corners = (None
                                     if self.upsample == 'nearest' else False)
                    upsample_feat = F.interpolate(
                        laterals[i],
                        scale_factor=2,
                        mode=self.upsample,
                        align_corners=align_corners)
                elif self.upsample == 'deconv':
                    upsample_feat = self.upsample_modules[i - 1](
                        laterals[i], output_size=laterals[i - 1].size())
                elif self.upsample == 'pixel_shuffle':
                    upsample_feat = self.upsample_modules[i - 1](laterals[i])
                    upsample_feat = F.pixel_shuffle(upsample_feat, 2)
                elif self.upsample == 'carafe':
                    upsample_feat = self.upsample_modules[i - 1](laterals[i])
                else:
                    AssertionError('upsample method not impl')
            else:
                upsample_feat = laterals[i]
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)
        return tuple(outs)

import torch.nn as nn
from mmcv.cnn import ConvModule, caffe2_xavier_init

from mmdet.ops.merge_cells import GlobalPoolingCell, SumCell
from ..builder import NECKS


@NECKS.register_module()
class NASFPN(nn.Module):
    """NAS-FPN.

    Implementation of `NAS-FPN: Learning Scalable Feature Pyramid Architecture
    for Object Detection <https://arxiv.org/abs/1904.07392>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): It decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 stack_times,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 norm_cfg=None):
        super(NASFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.stack_times = stack_times
        self.norm_cfg = norm_cfg

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
        self.add_extra_convs = add_extra_convs

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.lateral_convs.append(l_conv)

        # add extra downsample layers (stride-2 pooling or conv)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(
                out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
            self.extra_downsamples.append(
                nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))

        # add NAS FPN connections
        self.fpn_stages = nn.ModuleList()
        for _ in range(self.stack_times):
            stage = nn.ModuleDict()
            # gp(p6, p4) -> p4_1
            stage['gp_64_4'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_1, p4) -> p4_2
            stage['sum_44_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p4_2, p3) -> p3_out
            stage['sum_43_3'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p3_out, p4_2) -> p4_out
            stage['sum_34_4'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            stage['gp_43_5'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_55_5'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            stage['gp_54_7'] = GlobalPoolingCell(with_out_conv=False)
            stage['sum_77_7'] = SumCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            # gp(p7_out, p5_out) -> p6_out
            stage['gp_75_6'] = GlobalPoolingCell(
                in_channels=out_channels,
                out_channels=out_channels,
                out_norm_cfg=norm_cfg)
            self.fpn_stages.append(stage)

    def init_weights(self):
        """Initialize the weights of module"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        """Forward function"""
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build P6-P7 on top of P5
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        p3, p4, p5, p6, p7 = feats

        for stage in self.fpn_stages:
            # gp(p6, p4) -> p4_1
            p4_1 = stage['gp_64_4'](p6, p4, out_size=p4.shape[-2:])
            # sum(p4_1, p4) -> p4_2
            p4_2 = stage['sum_44_4'](p4_1, p4, out_size=p4.shape[-2:])
            # sum(p4_2, p3) -> p3_out
            p3 = stage['sum_43_3'](p4_2, p3, out_size=p3.shape[-2:])
            # sum(p3_out, p4_2) -> p4_out
            p4 = stage['sum_34_4'](p3, p4_2, out_size=p4.shape[-2:])
            # sum(p5, gp(p4_out, p3_out)) -> p5_out
            p5_tmp = stage['gp_43_5'](p4, p3, out_size=p5.shape[-2:])
            p5 = stage['sum_55_5'](p5, p5_tmp, out_size=p5.shape[-2:])
            # sum(p7, gp(p5_out, p4_2)) -> p7_out
            p7_tmp = stage['gp_54_7'](p5, p4_2, out_size=p7.shape[-2:])
            p7 = stage['sum_77_7'](p7, p7_tmp, out_size=p7.shape[-2:])
            # gp(p7_out, p5_out) -> p6_out
            p6 = stage['gp_75_6'](p7, p5, out_size=p6.shape[-2:])

        return p3, p4, p5, p6, p7

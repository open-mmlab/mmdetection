import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init

from ..builder import NECKS


def generate_coord(input_feat):
    x_range = torch.linspace(
        -1, 1, input_feat.shape[-1], device=input_feat.device)
    y_range = torch.linspace(
        -1, 1, input_feat.shape[-2], device=input_feat.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([input_feat.shape[0], 1, -1, -1])
    x = x.expand([input_feat.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat


@NECKS.register_module()
class SemanticFPN(nn.Module):
    """Implementation of Semantic FPN used in Panoptic FPN.

    Args:
        in_channels (int): Input channels.
        feat_channels (int): Feature channels in lateral convolutional layers.
        out_channels (int): Output channels of feature map.
        start_level (int): The first level of feature pyramid to be gathered.
        end_level (int): The last level of feature pyramid to be gathered.
        cat_coors (bool, optional): Whether to concatenate coordinates.
            Defaults to False.
        cat_coors_level (int, optional): The level to concatenate coordinates.
            Defaults to 3.
        return_list (bool, optional): Whether to return the feature map in
            a list. Defaults to False.
        upsample_times (int, optional): The number of times of upsampling.
            Defaults to 3.
        out_act_cfg (dict, optional): The activation config of output.
            Defaults to `dict(type='ReLU')`.
        conv_cfg (dict, optional): Config of convolutional layers.
            Defaults to None.
        norm_cfg (dict, optional): Config of normalization layers.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 start_level,
                 end_level,
                 cat_coors=False,
                 cat_coors_level=3,
                 return_list=False,
                 upsample_times=3,
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticFPN, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cat_coors = cat_coors
        self.cat_coors_level = cat_coors_level
        self.return_list = return_list
        self.upsample_times = upsample_times

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                if i == self.cat_coors_level and self.cat_coors:
                    chn = self.in_channels + 2
                else:
                    chn = self.in_channels
                if upsample_times == self.end_level - i:
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(i), one_conv)
                else:
                    for i in range(self.end_level - upsample_times):
                        one_conv = ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            stride=2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            inplace=False)
                        convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.cat_coors_level and self.cat_coors:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    if j < upsample_times - (self.end_level - i):
                        one_upsample = nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
                        convs_per_level.add_module('upsample' + str(j),
                                                   one_upsample)
                    continue

                one_conv = ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                if j < upsample_times - (self.end_level - i):
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j),
                                               one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = ConvModule(
            in_channels,
            self.out_channels,
            1,
            padding=0,
            conv_cfg=self.conv_cfg,
            act_cfg=out_act_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        mlvl_feats = []
        for i in range(self.start_level, self.end_level + 1):
            input_p = inputs[i]
            if i == self.cat_coors_level:
                if self.cat_coors:
                    coord_feat = generate_coord(input_p)
                    input_p = torch.cat([input_p, coord_feat], 1)

            mlvl_feats.append(self.convs_all_levels[i](input_p))

        feature_add_all_level = sum(mlvl_feats)
        feature_pred = self.conv_pred(feature_add_all_level)

        if self.return_list:
            return [feature_pred]
        else:
            return feature_pred


@NECKS.register_module()
class FusedSemanticMapper(nn.Module):
    r"""Multi-level fused pyramid mapper for semantic segmentation head
    used in Hybrid Task Cascade (HTC).

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |
        in_5 -> 1x1 conv ---

        Args:
            num_ins (int): The number of input feature levels.
            fusion_level (int): The expected level of the fused feature map.
            num_convs (int, optional): Number of convs to decode the feature.
                Defaults to 4.
            in_channels (int, optional): Input channels. Defaults to 256.
            out_channels (int, optional): Output channels. Defaults to 256.
            conv_cfg (dict, optional): Config of convolutional layers.
                Defaults to None.
            norm_cfg (dict, optional): Config of normalization layers.
                Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FusedSemanticMapper, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.normal_init = normal_init

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)

        return x

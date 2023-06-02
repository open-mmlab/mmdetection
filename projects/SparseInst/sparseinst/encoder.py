# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import caffe2_xavier_init, kaiming_init

from mmdet.registry import MODELS


class PyramidPoolingModule(nn.Module):

    def __init__(self,
                 in_channels,
                 channels=512,
                 sizes=(1, 2, 3, 6),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_channels + len(sizes) * channels,
                                    in_channels, 1)
        self.act = MODELS.build(act_cfg)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=self.act(stage(feats)),
                size=(h, w),
                mode='bilinear',
                align_corners=False) for stage in self.stages
        ] + [feats]
        out = self.act(self.bottleneck(torch.cat(priors, 1)))
        return out


@MODELS.register_module()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 with_ppm=True,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.with_ppm = with_ppm
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = nn.Conv2d(in_channel, self.num_channels, 1)
            output_conv = nn.Conv2d(
                self.num_channels, self.num_channels, 3, padding=1)
            caffe2_xavier_init(lateral_conv)
            caffe2_xavier_init(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        if self.with_ppm:
            self.ppm = PyramidPoolingModule(
                self.num_channels, self.num_channels // 4, act_cfg=act_cfg)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        kaiming_init(self.fusion)

    def forward(self, features):
        features = features[::-1]
        prev_features = self.fpn_laterals[0](features[0])
        if self.with_ppm:
            prev_features = self.ppm(prev_features)
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:],
                                                  self.fpn_laterals[1:],
                                                  self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [
            F.interpolate(x, size, mode='bilinear', align_corners=False)
            for x in outputs[1:]
        ]
        features = self.fusion(torch.cat(features, dim=1))
        return features

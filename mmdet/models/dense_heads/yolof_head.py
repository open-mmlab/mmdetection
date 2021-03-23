import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from torch.nn import BatchNorm2d, ReLU

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class YOLOFHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_num_convs=2,
                 reg_num_convs=4,
                 **kwargs):
        self.cls_num_convs = cls_num_convs
        self.reg_num_convs = reg_num_convs
        self.INF = 1e8
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(BatchNorm2d(self.in_channels))
            cls_subnet.append(ReLU())
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(BatchNorm2d(self.in_channels))
            bbox_subnet.append(ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_anchors,
            kernel_size=3,
            stride=1,
            padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) +
            torch.clamp(objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

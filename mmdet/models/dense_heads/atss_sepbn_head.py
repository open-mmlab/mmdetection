import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmdet.models.dense_heads import ATSSHead

from ..builder import HEADS


@HEADS.register_module()
class ATSSSepBNHead(ATSSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_ins,
                 **kwargs):
        self.num_ins = num_ins
        super(ATSSSepBNHead, self).__init__(num_classes,
                                            in_channels,
                                            **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                self.cls_convs[j][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[j][i].conv = self.reg_convs[0][i].conv

        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])


    def init_weights(self):
        for m in self.cls_convs:
            for mm in m:
                normal_init(mm.conv, std=0.01)
        for m in self.reg_convs:
            for mm in m:
                normal_init(mm.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_centerness, std=0.01)


    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        centernesses = []
        for i, x in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)

            cls_score = self.atss_cls(cls_feat)
            bbox_pred = self.scales[i](self.atss_reg(reg_feat)).float()
            centerness = self.atss_centerness(reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)

        return cls_scores, bbox_preds, centernesses
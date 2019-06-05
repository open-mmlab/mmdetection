import torch.nn as nn

from mmcv.cnn import normal_init
from .bbox_head import BBoxHead
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class RFCNHead(BBoxHead):

    def __init__(self,
                 psroipool_size,
                 in_channels,
                 conv_out_channels,
                 num_classes,
                 reg_class_agnostic,
                 target_means,
                 target_stds,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
        self.psroipool_size = psroipool_size
        self.reg_class_agnostic = reg_class_agnostic
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.conv_new = nn.Conv2d(in_channels, conv_out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_rfcn_cls = nn.Conv2d(
            conv_out_channels, psroipool_size * psroipool_size * num_classes,
            1)
        out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
        self.conv_rfcn_reg = nn.Conv2d(
            conv_out_channels, psroipool_size * psroipool_size * out_dim_reg,
            1)
        self.avepool = nn.AvgPool2d(psroipool_size)

    def init_weights(self):
        normal_init(self.conv_rfcn_cls, mean=0, std=0.01)
        normal_init(self.conv_rfcn_reg, mean=0, std=0.01)

    def forward(self, layer4_feat, rois, cls_roi_extractor, reg_roi_extractor):
        feat = self.relu(self.conv_new(layer4_feat))
        rfcn_cls = self.conv_rfcn_cls(feat)
        rfcn_reg = self.conv_rfcn_reg(feat)
        psroi_pooled_cls = cls_roi_extractor([rfcn_cls], rois)
        psroi_pooled_reg = reg_roi_extractor([rfcn_reg], rois)
        cls_score = self.avepool(psroi_pooled_cls)[:, :, 0, 0]
        bbox_pred = self.avepool(psroi_pooled_reg)[:, :, 0, 0]
        return cls_score, bbox_pred

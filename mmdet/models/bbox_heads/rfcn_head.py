import torch.nn as nn

from mmdet.ops import PSRoIPool
from .bbox_head import BBoxHead
from ..registry import HEADS


@HEADS.register_module
class RFCNHead(BBoxHead):

    def __init__(self, psroipool_size, in_channels, conv_out_channels,
                 num_classes, reg_class_agnostic, target_means, target_stds):
        super(RFCNHead, self).__init__()
        self.psroipool_size = psroipool_size
        self.reg_class_agnostic = reg_class_agnostic
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds

        self.conv_new = nn.Conv2d(in_channels, conv_out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_rfcn_cls = nn.Conv2d(
            conv_out_channels, psroipool_size * psroipool_size * num_classes,
            1)
        self.psroi_pooling_cls = PSRoIPool(psroipool_size, 1.0 / 16,
                                           psroipool_size)
        out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
        self.conv_rfcn_reg = nn.Conv2d(
            conv_out_channels, psroipool_size * psroipool_size * out_dim_reg,
            1)
        self.psroi_pooling_reg = PSRoIPool(psroipool_size, 1.0 / 16,
                                           psroipool_size)
        self.avepool = nn.AvgPool2d(psroipool_size)

    def init_weights(self):
        self.conv_rfcn_cls.weight.data.normal_(0, 0.01)
        self.conv_rfcn_cls.bias.data.zero_()
        self.conv_rfcn_reg.weight.data.normal_(0, 0.001)
        self.conv_rfcn_reg.bias.data.zero_()

    def forward(self, layer4_feat, rois):
        feat = self.relu(self.conv_new(layer4_feat))
        rfcn_cls = self.conv_rfcn_cls(feat)
        rfcn_reg = self.conv_rfcn_reg(feat)
        psroi_pooled_cls = self.psroi_pooling_cls(rfcn_cls, rois)
        psroi_pooled_reg = self.psroi_pooling_reg(rfcn_reg, rois)
        cls_score = self.avepool(psroi_pooled_cls)[:, :, 0, 0]
        bbox_pred = self.avepool(psroi_pooled_reg)[:, :, 0, 0]
        return cls_score, bbox_pred

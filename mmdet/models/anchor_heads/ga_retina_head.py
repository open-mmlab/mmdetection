import torch.nn as nn
from mmcv.cnn import normal_init

from .guided_anchor_head import GuidedAnchorHead
from mmdet.ops import DeformConv, MaskedConv2d
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class GARetinaHead(GuidedAnchorHead):

    def __init__(self, num_classes, in_channels, stacked_convs=4, **kwargs):
        self.stacked_convs = stacked_convs
        super(GARetinaHead, self).__init__(
            num_classes,
            in_channels,
            cls_sigmoid_loss=True,
            cls_focal_loss=True,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))
            self.reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, 3, stride=1, padding=1))

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        deformable_groups = 4
        offset_channels = 3 * 3 * 2
        self.conv_offset_cls = nn.Conv2d(
            self.num_anchors * 2,
            deformable_groups * offset_channels,
            1,
            bias=False)
        self.conv_adaption_cls = DeformConv(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            padding=1,
            deformable_groups=deformable_groups)
        self.conv_offset_reg = nn.Conv2d(
            self.num_anchors * 2,
            deformable_groups * offset_channels,
            1,
            bias=False)
        self.conv_adaption_reg = DeformConv(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            padding=1,
            deformable_groups=deformable_groups)
        self.retina_cls = MaskedConv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
            normal_init(m, std=0.01)

        normal_init(self.conv_offset_cls, std=0.1)
        normal_init(self.conv_adaption_cls, std=0.01)
        normal_init(self.conv_offset_reg, std=0.1)
        normal_init(self.conv_adaption_reg, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        offset_cls = self.conv_offset_cls(shape_pred.detach())
        cls_feat = self.relu(self.conv_adaption_cls(cls_feat, offset_cls))
        offset_reg = self.conv_offset_reg(shape_pred.detach())
        reg_feat = self.relu(self.conv_adaption_reg(reg_feat, offset_reg))

        if not x.requires_grad:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
            cls_score = self.retina_cls.forward_test(cls_feat, mask)
            bbox_pred = self.retina_reg.forward_test(reg_feat, mask)
        else:
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred, shape_pred, loc_pred

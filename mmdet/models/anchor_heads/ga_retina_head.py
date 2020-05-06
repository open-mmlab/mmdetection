import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.ops import MaskedConv2d
from ..builder import HEADS
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead


@HEADS.register_module()
class GARetinaHead(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(GARetinaHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        self.feature_adaption_cls = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.feature_adaption_reg = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.retina_cls = MaskedConv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        cls_feat = self.feature_adaption_cls(cls_feat, shape_pred)
        reg_feat = self.feature_adaption_reg(reg_feat, shape_pred)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.retina_cls(cls_feat, mask)
        bbox_pred = self.retina_reg(reg_feat, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

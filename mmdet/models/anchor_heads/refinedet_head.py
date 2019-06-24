import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import AnchorGenerator, anchor_target, multi_apply
from .anchor_head import AnchorHead
from ..losses import smooth_l1_loss
from ..registry import HEADS


# TODO: add loss evaluator for SSD
@HEADS.register_module
class RefineDetHead(AnchorHead):
    mbox = {
        '320': [3, 3, 3, 3],  # number of boxes per feature map location
        '512': [3, 3, 3, 3],  # number of boxes per feature map location
    }

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 512, 1024, 512),
                 anchor_strides=(8, 16, 32, 64),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(AnchorHead, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]

        # ARM
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(in_channels[i], num_anchors[i] * 4, kernel_size=3, padding=1))
            cls_convs.append(
                nn.Conv2d(in_channels[i], num_anchors[i] * 2, kernel_size=3, padding=1))
        self.arm_reg = nn.ModuleList(reg_convs)
        self.arm_cls = nn.ModuleList(cls_convs)

        # TCB
        TCB = self.add_tcb(in_channels)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        # ODM
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(256, num_anchors[i] * 4, kernel_size=3, padding=1))
            cls_convs.append(
                nn.Conv2d(256, num_anchors[i] * num_classes, kernel_size=3, padding=1))
        self.odm_reg = nn.ModuleList(reg_convs)
        self.odm_cls = nn.ModuleList(cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]  # 4 or 6 ratio
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):

        tcb_feats = list()
        arm_cls = list()
        arm_reg = list()
        odm_cls = list()
        odm_reg = list()

        # apply ARM to feats
        for feat, reg_conv, cls_conv in zip(feats, self.arm_reg,self.arm_cls):
            arm_cls.append(cls_conv(feat))
            arm_reg.append(reg_conv(feat))

        # calculate TCB features
        p = None
        for k, v in enumerate(feats[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3 - k) * 3 + i](s)
                # print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3 - k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3 - k) * 3 + i](s)
            p = s
            tcb_feats.append(s)

        tcb_feats.reverse()

        # apply ODM to feats
        for feat, reg_conv, cls_conv in zip(tcb_feats, self.arm_reg,self.arm_cls):
            odm_cls.append(cls_conv(feat))
            odm_reg.append(reg_conv(feat))
        
        return arm_cls, arm_reg, odm_cls, odm_reg

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self, arm_cls, arm_reg, odm_cls, odm_reg, gt_bboxes,
             gt_labels, img_metas, cfg, gt_bboxes_ignore=None):
        arm_criterion = refinedet_mutlibox_loss(2, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
        odm_criterion = refinedet_mutlibox_loss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, use_ARM=True)

        arm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_cls], 1)
        arm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in arm_reg], 1)

        odm_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_cls], 1)
        odm_reg = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1)
                             for o in odm_reg], 1)

        predict = (
            arm_reg.view(arm_reg.size(0), -1, 4),
            arm_cls.view(arm_cls.size(0), -1, 2),
            odm_reg.view(odm_reg.size(0), -1, 4),
            odm_cls.view(odm_cls.size(0), -1, self.num_classes)
        )


        arm_loss_l, arm_loss_c = arm_criterion(predict, targets)
        odm_loss_l, odm_loss_c = odm_criterion(preditc, targets)

        return dict(arm_reg_loss=arm_reg_loss,
                    arm_cls_loss=arm_cls_loss,
                    odm_reg_loss=odm_reg_loss,
                    odm_cls_loss=odm_cls_loss)

    def add_tcb(self, in_channels):
        feature_scale_layers = []
        feature_upsample_layers = []
        feature_pred_layers = []
        for k, v in enumerate(in_channels):
            feature_scale_layers += [nn.Conv2d(in_channels[k], 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
                                     ]
            feature_pred_layers += [nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(inplace=True)
                                    ]
            if k != len(in_channels) - 1:
                feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
        return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)



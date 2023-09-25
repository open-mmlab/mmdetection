# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .accuracy import accuracy
from .ecm_utils import get_detection_weight


@MODELS.register_module()
class EffectiveClassMarginLoss(nn.Module):

    def __init__(
        self,
        fg_bg_ratio=6.3313,  # empirical fg-bg ratio from Mask-RCNN.
        use_sigmoid=False,
        reduction='mean',
        loss_weight=1.0,
        num_classes=1203,
        dataset='lvis',
        **kwargs,
    ):
        super(EffectiveClassMarginLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.dataset = dataset

        if 'lvis' in self.dataset:
            n = torch.tensor(torch.load('lvis_v1_image_count.pkl')).float()
            print(n.shape)
        elif 'oid' in self.dataset:
            n = torch.tensor(torch.load('oid_image_count.pkl')).float()
        n = torch.cat([n, n.sum()[None] * fg_bg_ratio])

        self.register_buffer('sample_n', n)
        self.register_buffer('detection_cls_weight', get_detection_weight(n))

        self.custom_cls_channels = True
        self.custom_activation = True
        self.return_dict = True

    def forward(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):
        B = cls_score.shape[0]
        C = cls_score.shape[1] - 1
        inds = torch.arange(0, B, dtype=torch.long, device=cls_score.device)
        target = cls_score.new_zeros(B, C + 1)
        target[inds, labels] = 1  # B x (C + 1)

        pos_w, neg_w = self.compute_weight(cls_score)  # B x (C + 1)

        pos_w = pos_w * target.new_ones(B, C + 1)
        neg_w = neg_w * target.new_ones(B, C + 1)  # B x (C + 1)

        score_exp_pos = (cls_score + pos_w).exp()
        score_exp_neg = (-cls_score + neg_w).exp()  # B x (C + 1)
        pred_pos = score_exp_pos / (score_exp_pos + score_exp_neg)
        pred_neg = score_exp_neg / (score_exp_pos + score_exp_neg)
        loss_cls = -(pred_pos.log() * target + pred_neg.log() * (1 - target))

        cls_weight = self.detection_cls_weight.to(loss_cls.device)
        loss_cls = (loss_cls * cls_weight).sum() / B

        return loss_cls * self.loss_weight

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        acc = dict()
        obj_labels = (labels == self.num_classes).long()  # 0 fg, 1 bg
        acc_objectness = accuracy(
            torch.cat([1 - cls_score[:, -1:], cls_score[:, -1:]], dim=1),
            obj_labels)
        acc_classes = accuracy(cls_score[:, :-1][pos_inds], labels[pos_inds])

        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def get_cls_channels(self, num_classes):
        return num_classes + 1

    def get_activation(self, cls_score):
        cls_score = cls_score.sigmoid()
        cls_score[:, :self.num_classes] *= (1 -
                                            cls_score[:, self.num_classes:])

        return cls_score

    def compute_weight(self, cls_score):
        B, C = cls_score.shape

        n_pos = self.sample_n
        n_neg = self.sample_n.sum() - self.sample_n
        pos_w = (n_neg.pow(1 / 4) /
                 (n_pos.pow(1 / 4) + n_neg.pow(1 / 4))).pow(-1).log()
        neg_w = (n_pos.pow(1 / 4) /
                 (n_pos.pow(1 / 4) + n_neg.pow(1 / 4))).pow(-1).log()
        pos_w = pos_w.view(1, -1).expand(B, C)
        neg_w = neg_w.view(1, -1).expand(B, C)

        return pos_w, neg_w

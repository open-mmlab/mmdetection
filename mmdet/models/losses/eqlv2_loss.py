# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class EQLV2Loss(nn.Module):

    def __init__(self,
                 use_sigmoid: bool = True,
                 reduction: str = 'mean',
                 class_weight: Optional[Tensor] = None,
                 loss_weight: float = 1.0,
                 num_classes: int = 1203,
                 use_distributed: bool = False,
                 mu: float = 0.8,
                 alpha: float = 4.0,
                 gamma: int = 12,
                 vis_grad: bool = False,
                 test_with_obj: bool = True) -> None:
        """`Equalization Loss v2 <https://arxiv.org/abs/2012.08548>`_

        Args:
            use_sigmoid (bool): EQLv2 uses the sigmoid function to transform
                the predicted logits to an estimated probability distribution.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            class_weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            loss_weight (float, optional): The weight of the total EQLv2 loss.
                Defaults to 1.0.
            num_classes (int): 1203 for lvis v1.0, 1230 for lvis v0.5.
            use_distributed (bool, float): EQLv2 will calculate the gradients
                on all GPUs if there is any. Change to True if you are using
                distributed training. Default to False.
            mu (float, optional): Defaults to 0.8
            alpha (float, optional): A balance factor for the negative part of
                EQLV2 Loss. Defaults to 4.0.
            gamma (int, optional): The gamma for calculating the modulating
                factor. Defaults to 12.
            vis_grad (bool, optional): Default to False.
            test_with_obj (bool, optional): Default to True.

        Returns:
            None.
        """
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.mu = mu
        self.alpha = alpha
        self.gamma = gamma
        self.use_distributed = use_distributed

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos
        # gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))

        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

        print_log(
            f'build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}',
            logger='current',
            level=logging.DEBUG)

    def forward(self,
                cls_score: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[Tensor] = None) -> Tensor:
        """`Equalization Loss v2 <https://arxiv.org/abs/2012.08548>`_

        Args:
            cls_score (Tensor): The prediction with shape (N, C), C is the
                number of classes.
            label (Tensor): The ground truth label of the predicted target with
                shape (N, C), C is the number of classes.
            weight (Tensor, optional): The weight of loss for each prediction.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
           Tensor: The calculated loss
        """
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score, target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, pred):
        pred = torch.sigmoid(pred)
        n_i, n_c = pred.size()
        bg_score = pred[:, -1].view(n_i, 1)
        if self.test_with_obj:
            pred[:, :-1] *= (1 - bg_score)
        return pred

    def collect_grad(self, pred, target, weight):
        prob = torch.sigmoid(pred)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        if self.use_distributed:
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, pred):
        neg_w = torch.cat([self.map_func(self.pos_neg), pred.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .accuracy import accuracy
from .cross_entropy_loss import cross_entropy
from .utils import weight_reduce_loss


def seesaw_ce_loss(cls_score: Tensor,
                   labels: Tensor,
                   label_weights: Tensor,
                   cum_samples: Tensor,
                   num_classes: int,
                   p: float,
                   q: float,
                   eps: float,
                   reduction: str = 'mean',
                   avg_factor: Optional[int] = None) -> Tensor:
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (Tensor): The learning label of the prediction.
        label_weights (Tensor): Sample-wise loss weight.
        cum_samples (Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(
        loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    arXiv: https://arxiv.org/abs/2008.10032

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
             of softmax. Only False is supported.
        p (float, optional): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float, optional): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int, optional): The number of classes.
             Default to 1203 for LVIS v1 dataset.
        eps (float, optional): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method that reduces the loss to a
             scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        return_dict (bool, optional): Whether return the losses as a dict.
             Default to True.
    """

    def __init__(self,
                 use_sigmoid: bool = False,
                 p: float = 0.8,
                 q: float = 2.0,
                 num_classes: int = 1203,
                 eps: float = 1e-2,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_dict: bool = True) -> None:
        super().__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer(
            'cum_samples',
            torch.zeros(self.num_classes + 1, dtype=torch.float))

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def _split_cls_score(self, cls_score: Tensor) -> Tuple[Tensor, Tensor]:
        """split cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).

        Returns:
            Tuple[Tensor, Tensor]: The score for classes and objectness,
                 respectively
        """
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes: int) -> int:
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score: Tensor) -> Tensor:
        """Get custom activation of cls_score.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).

        Returns:
            Tensor: The custom activation of cls_score with shape
                 (N, C + 1).
        """
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        score_classes = F.softmax(cls_score_classes, dim=-1)
        score_objectness = F.softmax(cls_score_objectness, dim=-1)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = torch.cat([score_classes, score_neg], dim=-1)
        return scores

    def get_accuracy(self, cls_score: Tensor,
                     labels: Tensor) -> Dict[str, Tensor]:
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).
            labels (Tensor): The learning label of the prediction.

        Returns:
            Dict [str, Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def forward(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward function.

        Args:
            cls_score (Tensor): The prediction with shape (N, C + 2).
            labels (Tensor): The learning label of the prediction.
            label_weights (Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".

        Returns:
            Tensor | Dict [str, Tensor]:
                 if return_dict == False: The calculated loss |
                 if return_dict == True: The dict of calculated losses
                 for objectness and classes, respectively.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        obj_labels = (labels == self.num_classes).long()

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)

        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(
                cls_score_classes[pos_inds], labels[pos_inds],
                label_weights[pos_inds], self.cum_samples[:self.num_classes],
                self.num_classes, self.p, self.q, self.eps, reduction,
                avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(
            cls_score_objectness, obj_labels, label_weights, reduction,
            avg_factor)

        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_objectness'] = loss_cls_objectness
            loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes + loss_cls_objectness
        return loss_cls

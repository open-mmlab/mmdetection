import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss_focal

import pdb

# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
   
    targets = F.one_hot(target, num_classes=81)
    targets = targets[:,1:]
    pred_sigmoid = pred.sigmoid()
    targets = targets.type_as(pred)
    pt = (1 - pred_sigmoid) * targets + pred_sigmoid * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) *
                    (1 - targets)) * pt.pow(gamma)
    
    loss = modified_cross_entropy(
        pred_sigmoid, targets, reduction='none') * focal_weight

    loss = torch.clamp(loss, min=0)
    loss = weight_reduce_loss_focal(loss, weight, reduction, avg_factor)
    return loss

def modified_cross_entropy(pred, target, reduction ='none'):
    
    shifting_factor = -1*torch.ones(pred.shape)
    shifting_factor = torch.exp(shifting_factor)
    shifting_factor = shifting_factor.type_as(pred)
    
    return -1*(target*torch.log(pred+shifting_factor)+\
           (1-target)*torch.log(1-pred+shifting_factor))
    

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='none',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction) 
        if self.use_sigmoid: 
            loss_cls = py_sigmoid_focal_loss(
                    pred,
                    target,
                    weight,
                    gamma=self.gamma,
                    alpha=self.alpha,
                    reduction=reduction,
                    avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

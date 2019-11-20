import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
import pdb


def lrp_loss(pred, 
             label, 
             weight=None, 
             reduction='mean', 
             avg_factor=None,
             eps = 1e-6):
    
    
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))
    if weight is not None:
        weight = weight.float()
    
    pred_sigmoid = pred.sigmoid()
    label = label.type_as(pred)
    
    

    #print("Sigmoid activation:", pred_sigmoid[0])
    loss = 1*(label*torch.cos(1.57*pred_sigmoid)+\
               (1-label)*torch.cos(1.57*(1-pred_sigmoid)))
    #print("Loss max: {}, Loss min: {}\n".format(loss.max(), loss.min()))

    #loss = F.binary_cross_entropy_with_logits(
    #        pred, label.float(), weight, reduction='none')
    #print("Loss:",loss[0])
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    #print("Reduced Loss:",loss)
    pdb.set_trace()
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels>=1).squeeze()

    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


@LOSSES.register_module
class LRPLoss(nn.Module):

    def __init__(self,
                 use_sigmoid = True,
                 use_mask = False,
                 reduction = 'mean',
                 loss_weight=1.0):
        super(LRPLoss, self).__init__()
        
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * lrp_loss(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            eps = 1e-6,
            **kwargs)
        return loss_cls


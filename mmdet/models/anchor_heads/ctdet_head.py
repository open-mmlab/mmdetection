import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.core import auto_fp16
from ..registry import HEADS


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    # ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    ind = ind.expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegL1Loss(nn.Module):

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        # mask = mask.unsqueeze(2).expand_as(pred).float()
        mask = mask.expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CtdetLoss(torch.nn.Module):

    def __init__(self):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg_wh = RegL1Loss()

        self.num_stacks = 1
        self.wh_weight = 0.1
        self.off_weight = 1
        self.hm_weight = 1

    def forward(self, outputs, **kwargs):
        batch = kwargs
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.num_stacks):
            output = outputs[s]
            output['hm'] = torch.clamp(
                output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)

            hm_loss += self.crit(output['hm'], batch['hm']) / self.num_stacks
            if self.wh_weight > 0:
                wh_loss += self.crit_reg_wh(output['wh'], batch['reg_mask'],
                                            batch['ind'],
                                            batch['wh']) / self.num_stacks

            if self.off_weight > 0:
                off_loss += self.crit_reg_wh(output['reg'], batch['reg_mask'],
                                             batch['ind'],
                                             batch['reg']) / self.num_stacks

        losses = {
            'hm_loss': self.hm_weight * hm_loss,
            'wh_loss': self.wh_weight * wh_loss,
            'off_loss': self.off_weight * off_loss
        }
        return losses


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@HEADS.register_module
class CtdetHead(nn.Module):
    """Simplest ctdet head with focal loss for heatmap"""

    def __init__(self,
                 heads,
                 loss_ctdet=dict(type='CtdetLoss'),
                 head_conv=256,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(CtdetHead, self).__init__()
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(
                        64, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=1,
                        stride=1,
                        padding=1 // 2,
                        bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    64,
                    classes,
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self.loss_ctdet = CtdetLoss()

    def init_weights(self):
        print('initializing head weights')

    @auto_fp16()
    def forward(self, x):
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return [z]

    @auto_fp16()
    def loss(self, outs, **kwargs):
        return self.loss_ctdet(outs, **kwargs)

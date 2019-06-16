# https://github.com/zjhuang22/maskscoring_rcnn
import torch
import torch.nn as nn

from .htc_mask_head import HTCMaskHead
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target


def l2_loss(input, target):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    pos_inds = torch.nonzero(target > 0.0).squeeze(1)
    if pos_inds.shape[0] > 0:
        cond = torch.abs(input[pos_inds] - target[pos_inds])
        loss = 0.5 * cond**2 / pos_inds.shape[0]
    else:
        loss = input * 0.0
    return loss.sum()


@HEADS.register_module
class HTCMaskScoringHead(HTCMaskHead):

    def __init__(self, *arg, **kwargs):
        super(HTCMaskScoringHead, self).__init__(*arg, **kwargs)
        self.roi_feat = None
        self.mask_ratios = None
        self.mask_iou_head = nn.Sequential()
        padding = (self.conv_kernel_size - 1) // 2
        for i in range(3):
            in_channels = (
                self.conv_out_channels+1 if i == 0 else self.conv_out_channels)
            self.mask_iou_head.add_module(
                'conv{}'.format(i),
                ConvModule(in_channels,
                           self.conv_out_channels,
                           self.conv_kernel_size,
                           padding=padding,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg))
        self.mask_iou_head.add_module(
            'downsample',
            ConvModule(self.conv_out_channels,
                       self.conv_out_channels,
                       self.conv_kernel_size,
                       stride=2,
                       padding=padding,
                       conv_cfg=self.conv_cfg,
                       norm_cfg=self.norm_cfg))

        self.mask_iou_head.add_module(
            'global_downsample',
            ConvModule(self.conv_out_channels, 1024, 7, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.mask_iou_head.add_module(
            'fc1',
            ConvModule(1024, 1024, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.mask_iou_head.add_module(
            'fc2',
            ConvModule(1024, self.num_classes, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.mask_iou_pool = nn.MaxPool2d(2, 2)

    def forward(self, x, *arg, **kwargs):
        self.roi_feat = x
        super(HTCMaskScoringHead, self).forward(x, *arg, **kwargs)

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets, mask_ratios = mask_target(pos_proposals, pos_assigned_gt_inds,
                                                gt_masks, rcnn_train_cfg, if_mask_iou=True)
        self.mask_ratios = mask_ratios
        return mask_targets  # denote only positive propose will be fed in mask head

    def loss(self, mask_pred, mask_targets, labels):
        """

        :param mask_pred:    [n_pos_roi, n_cls, h, w]
        :param mask_targets: [n_pos_roi, h, w], {0, 1}
        :param labels:       [n_pos_roi], int
        :return:
        """
        assert self.roi_feat, 'must forward once before compute loss'
        assert self.mask_ratios, 'must get_target once before compute loss'

        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        num_rois = mask_pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
        mask_pred = mask_pred[inds, labels]  # [n_pos_roi, h, w]
        pred_slice_pooled = self.mask_iou_pool(mask_pred)
        mask_iou = self.mask_iou_head(torch.cat([self.roi_feat, pred_slice_pooled], dim=1))  # [n_pos_roi, n_cls, 1, 1]
        mask_iou = mask_iou.squeeze()  # [n_pos_roi, n_cls]
        mask_iou = mask_iou[inds, labels]   # [n_pos_roi]

        # mask_iou为bbox中的交集 / 整个图像中的并集
        mask_pred[:] = mask_pred > 0  # inplace change
        mask_ovr = mask_pred * mask_targets
        mask_ovr_area = mask_ovr.sum(dim=[1, 2])
        mask_targets_full_area = mask_targets.sum(dim=[1, 2]) / self.mask_ratios
        mask_union_area = mask_pred.sum(dim=[1, 2]) + mask_targets_full_area - mask_ovr_area

        value_0 = torch.zeros(mask_pred.shape[0], device=labels.device)
        value_1 = torch.ones(mask_pred.shape[0], device=labels.device)
        mask_ovr_area = torch.max(mask_ovr_area, value_0)
        mask_union_area = torch.max(mask_union_area, value_1)
        mask_iou_targets = mask_ovr_area / mask_union_area
        mask_iou_targets = mask_iou_targets.detach()  # [n_pos_roi]

        loss_mask_iou = l2_loss(mask_iou, mask_iou_targets)
        loss['loss_mask_iou'] = loss_mask_iou

        return loss


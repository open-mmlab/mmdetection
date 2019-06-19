import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..registry import HEADS


@HEADS.register_module
class MaskIoUHead(nn.Module):
    """Mask IoU Head.

    Mask IoU Head takes both mask features and predicted mask to regress
    the mask IoU, which calibrates the cls score to sort segm results during
    evaluation.
    """

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=81):
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = (
                self.in_channels + 1 if i == 0 else self.conv_out_channels)
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))

        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = self.conv_out_channels * (
                roi_feat_size // 2)**2 if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.mask_iou = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.mask_iou, std=0.01)

    def forward(self, x, mask_pred):
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))
        x = torch.cat((x, mask_pred_pooled), 1)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.mask_iou(x)
        return mask_iou

    def get_target(self, sampling_results, gt_masks, mask_pred, mask_targets,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        return self.mask_iou_target(pos_proposals, pos_assigned_gt_inds,
                                    gt_masks, mask_pred, mask_targets,
                                    rcnn_train_cfg)

    def loss(self, mask_iou_pred, mask_iou_targets):
        loss = dict()
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss['loss_mask_iou'] = 0.5 * F.mse_loss(
                mask_iou_pred[pos_inds], mask_iou_targets[pos_inds])[None]
        else:
            loss['loss_mask_iou'] = mask_iou_pred * 0
        return loss

    def mask_iou_target(self, pos_proposals_list, pos_assigned_gt_inds_list,
                        gt_masks_list, mask_preds, mask_targets, cfg):
        """Compute mask IoU target of each positive proposal.

        Mask IoU target is the IoU of the predicted mask and the gt mask of
        corresponding gt instance. GT masks are of the same size with image,
        so firstly compute the area ratio of the gt mask inside the proposal
        and the gt mask of the corresponding instance, and use it to compute
        the full area of the instance at the scale of mask pred.

        Args:
            pos_proposals_list (list[Tensor]): Positive proposals of each
                image, shape of each pos proposals (n, 4).
            pos_assigned_gt_inds_list (list[Tensor]): Positive assigned gt
                inds of each image, shape of each pos_assigned_gt_inds (n,).
            gt_masks_list (list[ndarray]): Ground truth mask of each image,
                shape of gt_masks (num_obj, img_h, img_w).
            mask_preds (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Target mask of each positive proposal,
                shape (num_pos, h, w).

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        area_ratios = map(self.get_area_ratio, pos_proposals_list,
                          pos_assigned_gt_inds_list, gt_masks_list)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_pred = (mask_preds > cfg.mask_thr_binary).float()

        # mask target is either 0 or 1
        mask_overlaps = (mask_pred * mask_targets).sum((-1, -2))

        # mask area of the whole instance
        full_areas = mask_targets.sum((-1, -2)) / area_ratios

        mask_unions = mask_pred.sum((-1, -2)) + full_areas - mask_overlaps

        mask_iou_targets = mask_overlaps / mask_unions
        return mask_iou_targets

    def get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance"""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # only sum gt mask once to save time
            gt_instance_mask_area = gt_masks.sum((-1, -2))
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)

                # crop the gt mask inside the proposal
                gt_mask_in_proposal = gt_mask[y1:y1 + h, x1:x1 + w]

                ratio = gt_mask_in_proposal.sum() / (
                    gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7
                )  # avoid zero
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device) + 1e-7  # avoid zero
        else:
            area_ratios = pos_proposals.new_zeros((0, ))
        return area_ratios

    def get_mask_scores(self, mask_feats, mask_pred, det_bboxes, det_labels):
        inds = range(det_labels.size(0))
        mask_ious = self.forward(mask_feats, mask_pred[inds, det_labels + 1])
        mask_scores = mask_ious[inds, det_labels + 1] * det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [
            mask_scores[det_labels == i] for i in range(self.num_classes - 1)
        ]

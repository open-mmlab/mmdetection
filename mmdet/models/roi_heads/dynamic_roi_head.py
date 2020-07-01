import numpy as np
import torch

from mmdet.core import bbox2roi
from mmdet.models.losses import SmoothL1Loss
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class DynamicRoIHead(StandardRoIHead):
    """RoI head for `Dynamic R-CNN <https://arxiv.org/abs/2004.06002>`_."""

    def __init__(self, **kwargs):
        super(DynamicRoIHead, self).__init__(**kwargs)
        assert isinstance(self.bbox_head.loss_bbox, SmoothL1Loss)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history = []

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cur_iou = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # record the `iou_topk`-th largest IoU in an image
                iou_topk = min(self.train_cfg.dynamic_rcnn.iou_topk,
                               len(assign_result.max_overlaps))
                ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                cur_iou.append(ious[-1].item())
                sampling_results.append(sampling_result)
            # average the current IoUs over images
            cur_iou = np.mean(cur_iou)
            self.iou_history.append(cur_iou)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        # update IoU threshold and SmoothL1 beta
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            new_iou_thr, new_beta = self.update_hyperparameters()

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        num_imgs = len(img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # record the `beta_topk`-th smallest target
        # `bbox_targets[2]` and `bbox_targets[3]` stand for bbox_targets
        # and bbox_weights, respectively
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
        beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                        num_pos)
        cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
        self.beta_history.append(cur_target)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def update_hyperparameters(self):
        """
        Update hyperparameters like `iou_thr` and `SmoothL1 beta` based
        on the training statistics.

        Returns:
            tuple[float]: the updated `iou_thr` and `SmoothL1 beta`
        """
        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history))
        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr
        new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta,
                       np.median(self.beta_history))
        self.beta_history = []
        self.bbox_head.loss_bbox.beta = new_beta
        return new_iou_thr, new_beta

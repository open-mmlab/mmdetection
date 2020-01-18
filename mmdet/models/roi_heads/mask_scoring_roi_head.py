import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import HEADS
from .base_roi_head import BaseRoIHead


@HEADS.register_module
class MaskScoringRoIHead(BaseRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self,
                 mask_iou_head,
                 **kwargs):
        super(MaskScoringRoIHead, self).__init__(**kwargs)
        assert mask_iou_head is not None
        self.mask_iou_head = builder.build_head(mask_iou_head)
        self.mask_iou_head.init_weights()

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_meta):
        mask_feats = self.extract_mask_feats(x, sampling_results, bbox_feats)

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)

            # mask iou head forward and loss
            pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
            mask_iou_pred = self.mask_iou_head(mask_feats, pos_mask_pred)
            pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                              pos_labels]
            mask_iou_targets = self.mask_iou_head.get_target(
                sampling_results, gt_masks, pos_mask_pred, mask_targets,
                self.train_cfg)
            loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                    mask_iou_targets)
            loss_mask.update(loss_mask_iou)
            return loss_mask
        else:
            return None   

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
            mask_scores = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg,
                                                       ori_shape, scale_factor,
                                                       rescale)
            # get mask scores with mask iou head
            mask_iou_pred = self.mask_iou_head(
                mask_feats, mask_pred[range(det_labels.size(0)),
                                      det_labels + 1])
            mask_scores = self.mask_iou_head.get_mask_scores(
                mask_iou_pred, det_bboxes, det_labels)
        return segm_result, mask_scores


# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import HEADS
from .double_roi_head import DoubleHeadRoIHead
from mmdet.core import bbox2roi


@HEADS.register_module()
class DoubleHeadRoIHeadExt(DoubleHeadRoIHead):

    def __init__(self, **kwargs):
        super(DoubleHeadRoIHeadExt, self).__init__(**kwargs)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['fc_cls_score'],
                                        bbox_results['fc_bbox_pred'],
                                        bbox_results['conv_cls_score'],
                                        bbox_results['conv_bbox_pred'],
                                        rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing time."""
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        fc_cls_score, fc_bbox_pred, conv_cls_score, conv_bbox_pred = self.bbox_head(
            bbox_cls_feats, bbox_reg_feats)

        if not self.training:
            # Complementary Fusion of Classifiers
            cls_score = fc_cls_score + conv_cls_score * (1 - fc_cls_score)
            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=conv_bbox_pred,
                bbox_feats=bbox_cls_feats)
            return bbox_results

        bbox_results = dict(
            fc_cls_score=fc_cls_score,
            fc_bbox_pred=fc_bbox_pred,
            conv_cls_score=conv_cls_score,
            conv_bbox_pred=conv_bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results

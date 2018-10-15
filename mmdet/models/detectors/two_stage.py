import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import sample_bboxes, bbox2roi, bbox2result, multi_apply


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_rpn_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=None,
                      proposals=None):
        losses = dict()

        x = self.extract_feat(img)

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_bbox:
            (pos_proposals, neg_proposals, pos_assigned_gt_inds, pos_gt_bboxes,
             pos_gt_labels) = multi_apply(
                 sample_bboxes,
                 proposal_list,
                 gt_bboxes,
                 gt_bboxes_ignore,
                 gt_labels,
                 cfg=self.train_cfg.rcnn)
            (labels, label_weights, bbox_targets,
             bbox_weights) = self.bbox_head.get_bbox_target(
                 pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
                 self.train_cfg.rcnn)

            rois = bbox2roi([
                torch.cat([pos, neg], dim=0)
                for pos, neg in zip(pos_proposals, neg_proposals)
            ])
            # TODO: a more flexible way to configurate feat maps
            roi_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, labels,
                                            label_weights, bbox_targets,
                                            bbox_weights)
            losses.update(loss_bbox)

        if self.with_mask:
            mask_targets = self.mask_head.get_mask_target(
                pos_proposals, pos_assigned_gt_inds, gt_masks,
                self.train_cfg.rcnn)
            pos_rois = bbox2roi(pos_proposals)
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            torch.cat(pos_gt_labels))
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta,
            self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

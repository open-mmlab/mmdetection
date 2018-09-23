import torch
import torch.nn as nn

from .base import Detector
from .testing_mixins import RPNTestMixin, BBoxTestMixin
from .. import builder
from mmdet.core import bbox2roi, bbox2result, sample_proposals


class TwoStageDetector(Detector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Detector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        self.with_neck = True if neck is not None else False
        if self.with_neck:
            self.neck = builder.build_neck(neck)

        self.with_rpn = True if rpn_head is not None else False
        if self.with_rpn:
            self.rpn_head = builder.build_rpn_head(rpn_head)

        self.with_bbox = True if bbox_head is not None else False
        if self.with_bbox:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
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
                      proposals=None):
        losses = dict()

        x = self.extract_feat(img)

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)

        else:
            proposal_list = proposals

        (pos_inds, neg_inds, pos_proposals, neg_proposals,
         pos_assigned_gt_inds,
         pos_gt_bboxes, pos_gt_labels) = sample_proposals(
             proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels,
             self.train_cfg.rcnn)

        labels, label_weights, bbox_targets, bbox_weights = \
            self.bbox_head.get_bbox_target(
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

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        if proposals is None:
            proposals = self.simple_test_rpn(x, img_meta)
        if self.with_bbox:
            # BUG proposals shape?
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, [proposals], rescale=rescale)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.bbox_head.num_classes)
            return bbox_result
        else:
            proposals[:, :4] /= img_meta['scale_factor'].float()
            return proposals.cpu().numpy()

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        proposals = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.rpn_test_cfg)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposals, self.rcnn_test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0]['shape_scale'][0][-1]
        bbox_result = bbox2result(_det_bboxes, det_labels,
                                  self.bbox_head.num_classes)
        return bbox_result

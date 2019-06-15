from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler

import torch
import torch.nn as nn

@DETECTORS.register_module
class DoubleHeadsFasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(DoubleHeadsFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


    def forward_train(self,
                     img,
                     img_meta,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     gt_masks=None,
                     proposals=None):
       x = self.extract_feat(img)

       losses = dict()

       # RPN forward and loss
       if self.with_rpn:
           rpn_outs = self.rpn_head(x)
           rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                         self.train_cfg.rpn)
           rpn_losses = self.rpn_head.loss(
               *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
           losses.update(rpn_losses)

           proposal_cfg = self.train_cfg.get('rpn_proposal',
                                             self.test_cfg.rpn)
           proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
           proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
       else:
           proposal_list = proposals

       # assign gts and sample proposals
       if self.with_bbox or self.with_mask:
           bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
           bbox_sampler = build_sampler(
               self.train_cfg.rcnn.sampler, context=self)
           num_imgs = img.size(0)
           if gt_bboxes_ignore is None:
               gt_bboxes_ignore = [None for _ in range(num_imgs)]
           sampling_results = []
           for i in range(num_imgs):
               assign_result = bbox_assigner.assign(
                   proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                   gt_labels[i])
               sampling_result = bbox_sampler.sample(
                   assign_result,
                   proposal_list[i],
                   gt_bboxes[i],
                   gt_labels[i],
                   feats=[lvl_feat[i][None] for lvl_feat in x])
               sampling_results.append(sampling_result)

       # bbox head forward and loss
       if self.with_bbox:
           rois = bbox2roi([res.bboxes for res in sampling_results])
           # TODO: a more flexible way to decide which feature maps to use
           bbox_feats = self.bbox_roi_extractor(
               x[:self.bbox_roi_extractor.num_inputs], rois)
           if self.with_shared_head:
               bbox_feats = self.shared_head(bbox_feats)
           # cls_score, bbox_pred = self.bbox_head(bbox_feats)
           conv_cls_score, conv_bbox_pred, fc_cls_score, fc_bbox_pred = self.bbox_head(bbox_feats)

           bbox_targets = self.bbox_head.get_target(
               sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
           loss_bbox = self.bbox_head.loss(conv_cls_score, conv_bbox_pred, fc_cls_score, fc_bbox_pred,
                                           *bbox_targets)
           losses.update(loss_bbox)

       # mask head forward and loss
       if self.with_mask:
           if not self.share_roi_extractor:
               pos_rois = bbox2roi(
                   [res.pos_bboxes for res in sampling_results])
               mask_feats = self.mask_roi_extractor(
                   x[:self.mask_roi_extractor.num_inputs], pos_rois)
               if self.with_shared_head:
                   mask_feats = self.shared_head(mask_feats)
           else:
               pos_inds = []
               device = bbox_feats.device
               for res in sampling_results:
                   pos_inds.append(
                       torch.ones(
                           res.pos_bboxes.shape[0],
                           device=device,
                           dtype=torch.uint8))
                   pos_inds.append(
                       torch.zeros(
                           res.neg_bboxes.shape[0],
                           device=device,
                           dtype=torch.uint8))
               pos_inds = torch.cat(pos_inds)
               mask_feats = bbox_feats[pos_inds]
           mask_pred = self.mask_head(mask_feats)

           mask_targets = self.mask_head.get_target(
               sampling_results, gt_masks, self.train_cfg.rcnn)
           pos_labels = torch.cat(
               [res.pos_gt_labels for res in sampling_results])
           loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                           pos_labels)
           losses.update(loss_mask)

       return losses
   

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

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



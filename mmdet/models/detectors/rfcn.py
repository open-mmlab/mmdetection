from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS


@DETECTORS.register_module
class RFCN(BaseDetector, RPNTestMixin):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 cls_roi_extractor,
                 reg_roi_extractor,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(RFCN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.rpn_head = builder.build_head(rpn_head)
        self.bbox_head = builder.build_head(bbox_head)
        self.cls_roi_extractor = builder.build_roi_extractor(cls_roi_extractor)
        self.reg_roi_extractor = builder.build_roi_extractor(reg_roi_extractor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(RFCN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.rpn_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        return self.backbone(img)

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        layer3_feat, layer4_feat = self.extract_feat(img)

        losses = dict()
        rpn_outs = self.rpn_head([layer3_feat])
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta, self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=layer4_feat)
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        cls_score, bbox_pred = self.bbox_head(layer4_feat, rois,
                                              self.cls_roi_extractor,
                                              self.reg_roi_extractor)
        bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes,
                                                 gt_labels,
                                                 self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        losses.update(loss_bbox)
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        layer3_feat, layer4_feat = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            [layer3_feat], img_meta,
            self.test_cfg.rpn) if proposals is None else proposals

        rois = bbox2roi(proposal_list)
        cls_score, bbox_pred = self.bbox_head(layer4_feat, rois,
                                              self.cls_roi_extractor,
                                              self.reg_roi_extractor)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

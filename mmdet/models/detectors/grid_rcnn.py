from .two_stage import TwoStageDetector
from ..registry import DETECTORS

import torch

from .. import builder
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler


@DETECTORS.register_module
class GridRCNN(TwoStageDetector):
    """Grid R-CNN.

    This detector is the implementation of:
    - Grid R-CNN (https://arxiv.org/abs/1811.12030)
    - Grid R-CNN Plus: Faster and Better (https://arxiv.org/abs/1906.05688)
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 grid_roi_extractor,
                 grid_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        assert grid_head is not None
        super(GridRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if grid_roi_extractor is not None:
            self.grid_roi_extractor = builder.build_roi_extractor(
                grid_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.grid_roi_extractor = self.bbox_roi_extractor
        self.grid_head = builder.build_head(grid_head)

        self.init_extra_weights()

    def init_extra_weights(self):
        self.grid_head.init_weights()
        if not self.share_roi_extractor:
            self.grid_roi_extractor.init_weights()

    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_bboxes = new_bboxes
        return sampling_results

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

        if self.with_bbox:
            # assign gts and sample proposals
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
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
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

            # Grid head forward and loss
            sampling_results = self._random_jitter(sampling_results, img_meta)
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            grid_feats = self.grid_roi_extractor(
                x[:self.grid_roi_extractor.num_inputs], pos_rois)
            if self.with_shared_head:
                grid_feats = self.shared_head(grid_feats)
            # Accelerate training
            max_sample_num_grid = self.train_cfg.rcnn.get('max_num_grid', 192)
            sample_idx = torch.randperm(
                grid_feats.shape[0])[:min(grid_feats.
                                          shape[0], max_sample_num_grid)]
            grid_feats = grid_feats[sample_idx]

            grid_pred = self.grid_head(grid_feats)

            grid_targets = self.grid_head.get_target(sampling_results,
                                                     self.train_cfg.rcnn)
            grid_targets = grid_targets[sample_idx]

            loss_grid = self.grid_head.loss(grid_pred, grid_targets)
            losses.update(loss_grid)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=False)

        # pack rois into bboxes
        grid_rois = bbox2roi([det_bboxes[:, :4]])
        grid_feats = self.grid_roi_extractor(
            x[:len(self.grid_roi_extractor.featmap_strides)], grid_rois)
        if grid_rois.shape[0] != 0:
            self.grid_head.test_mode = True
            grid_pred = self.grid_head(grid_feats)
            det_bboxes = self.grid_head.get_bboxes(det_bboxes,
                                                   grid_pred['fused'],
                                                   img_meta)
            if rescale:
                det_bboxes[:, :4] /= img_meta[0]['scale_factor']
        else:
            det_bboxes = torch.Tensor([])

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        return bbox_results

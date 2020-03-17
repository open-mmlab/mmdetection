import torch

from mmdet.core import bbox2result, bbox2roi
from .. import builder
from ..registry import HEADS
from .base_roi_head import BaseRoIHead


@HEADS.register_module
class GridRoIHead(BaseRoIHead):
    """Grid roi head for Grid R-CNN.

    https://arxiv.org/abs/1811.12030
    """

    def __init__(self, grid_roi_extractor, grid_head, **kwargs):
        assert grid_head is not None
        super(GridRoIHead, self).__init__(**kwargs)
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

    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)

        # grid head
        grid_rois = rois[:100]
        grid_feats = self.grid_roi_extractor(
            x[:self.grid_roi_extractor.num_inputs], grid_rois)
        if self.with_shared_head:
            grid_feats = self.shared_head(grid_feats)
        grid_pred = self.grid_head(grid_feats)
        outs = outs + (grid_pred, )

        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_meta):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes,
                                                 gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *bbox_targets)

        # Grid head forward and loss
        sampling_results = self._random_jitter(sampling_results, img_meta)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        grid_feats = self.grid_roi_extractor(
            x[:self.grid_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
            grid_feats = self.shared_head(grid_feats)
        # Accelerate training
        max_sample_num_grid = self.train_cfg.get('max_num_grid', 192)
        sample_idx = torch.randperm(
            grid_feats.shape[0])[:min(grid_feats.shape[0], max_sample_num_grid
                                      )]
        grid_feats = grid_feats[sample_idx]

        grid_pred = self.grid_head(grid_feats)

        grid_targets = self.grid_head.get_target(sampling_results,
                                                 self.train_cfg)
        grid_targets = grid_targets[sample_idx]

        loss_grid = self.grid_head.loss(grid_pred, grid_targets)
        loss_bbox.update(loss_grid)
        return loss_bbox, bbox_feats

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg, rescale=False)
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

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

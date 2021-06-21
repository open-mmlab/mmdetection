import numpy as np
import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class GridRoIHead(StandardRoIHead):
    """Grid roi head for Grid R-CNN.

    https://arxiv.org/abs/1811.12030
    """

    def __init__(self, grid_roi_extractor, grid_head, **kwargs):
        assert grid_head is not None
        super(GridRoIHead, self).__init__(**kwargs)
        if grid_roi_extractor is not None:
            self.grid_roi_extractor = build_roi_extractor(grid_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.grid_roi_extractor = self.bbox_roi_extractor
        self.grid_head = build_head(grid_head)

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
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])

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
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        bbox_results = super(GridRoIHead,
                             self)._bbox_forward_train(x, sampling_results,
                                                       gt_bboxes, gt_labels,
                                                       img_metas)

        # Grid head forward and loss
        sampling_results = self._random_jitter(sampling_results, img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        # GN in head does not support zero shape input
        if pos_rois.shape[0] == 0:
            return bbox_results

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

        grid_targets = self.grid_head.get_targets(sampling_results,
                                                  self.train_cfg)
        grid_targets = grid_targets[sample_idx]

        loss_grid = self.grid_head.loss(grid_pred, grid_targets)

        bbox_results['loss_bbox'].update(loss_grid)
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=False)
        # pack rois into bboxes
        grid_rois = bbox2roi([det_bbox[:, :4] for det_bbox in det_bboxes])
        if grid_rois.shape[0] != 0:
            grid_feats = self.grid_roi_extractor(
                x[:len(self.grid_roi_extractor.featmap_strides)], grid_rois)
            self.grid_head.test_mode = True
            grid_pred = self.grid_head(grid_feats)
            # split batch grid head prediction back to each image
            num_roi_per_img = tuple(len(det_bbox) for det_bbox in det_bboxes)
            grid_pred = {
                k: v.split(num_roi_per_img, 0)
                for k, v in grid_pred.items()
            }

            # apply bbox post-processing to each image individually
            bbox_results = []
            num_imgs = len(det_bboxes)
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    bbox_results.append([
                        np.zeros((0, 5), dtype=np.float32)
                        for _ in range(self.bbox_head.num_classes)
                    ])
                else:
                    det_bbox = self.grid_head.get_bboxes(
                        det_bboxes[i], grid_pred['fused'][i], [img_metas[i]])
                    if rescale:
                        det_bbox[:, :4] /= img_metas[i]['scale_factor']
                    bbox_results.append(
                        bbox2result(det_bbox, det_labels[i],
                                    self.bbox_head.num_classes))
        else:
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head.num_classes)
            ] for _ in range(len(det_bboxes))]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

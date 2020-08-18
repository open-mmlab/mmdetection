import logging
import sys

import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    bbox_semaphore=None,
                                    global_lock=None):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape,
                    scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result

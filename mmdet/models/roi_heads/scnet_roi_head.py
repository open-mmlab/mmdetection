import numpy as np
import torch
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SCNetRoIHead(CascadeRoIHead):
    """RoIHead for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list): loss weight of cascade stages.
        semantic_roi_extractor (dict): config to init semantic roi extractor.
        semantic_head (dict): config to init semantic head.
        feat_relay_head (dict): config to init feature_relay_head.
        glbctx_head (dict): config to init global context head.
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 feat_relay_head=None,
                 glbctx_head=None,
                 **kwargs):
        super(SCNetRoIHead, self).__init__(num_stages, stage_loss_weights,
                                           **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            self.semantic_roi_extractor = build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        if feat_relay_head is not None:
            self.feat_relay_head = build_head(feat_relay_head)

        if glbctx_head is not None:
            self.glbctx_head = build_head(glbctx_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.mask_head = build_head(mask_head)

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    @property
    def with_feat_relay(self):
        """bool: whether the head has feature relay head"""
        return (hasattr(self, 'feat_relay_head')
                and self.feat_relay_head is not None)

    @property
    def with_glbctx(self):
        """bool: whether the head has global context head"""
        return hasattr(self, 'glbctx_head') and self.glbctx_head is not None

    def _fuse_glbctx(self, roi_feats, glbctx_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]
        return fused_feats

    def _slice_pos_feats(self, feats, sampling_results):
        """Get features from pos rois."""
        num_rois = [res.bboxes.size(0) for res in sampling_results]
        num_pos_rois = [res.pos_bboxes.size(0) for res in sampling_results]
        inds = torch.zeros(sum(num_rois), dtype=torch.bool)
        start = 0
        for i in range(len(num_rois)):
            start = 0 if i == 0 else start + num_rois[i - 1]
            stop = start + num_pos_rois[i]
            inds[start:stop] = 1
        sliced_feats = feats[inds]
        return sliced_feats

    def _bbox_forward(self,
                      stage,
                      x,
                      rois,
                      semantic_feat=None,
                      glbctx_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and semantic_feat is not None:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)
        cls_score, bbox_pred, relayed_feat = bbox_head(
            bbox_feats, return_shared_feat=True)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            relayed_feat=relayed_feat)
        return bbox_results

    def _mask_forward(self,
                      x,
                      rois,
                      semantic_feat=None,
                      glbctx_feat=None,
                      relayed_feat=None):
        """Mask head forward function used in both training and testing."""
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], rois)
        if self.with_semantic and semantic_feat is not None:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.with_glbctx and glbctx_feat is not None:
            mask_feats = self._fuse_glbctx(mask_feats, glbctx_feat, rois)
        if self.with_feat_relay and relayed_feat is not None:
            mask_feats = mask_feats + relayed_feat
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred)

        return mask_results

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None,
                            glbctx_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage,
            x,
            rois,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'], rois,
                                   *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None,
                            glbctx_feat=None,
                            relayed_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(
            x,
            pos_rois,
            semantic_feat=semantic_feat,
            glbctx_feat=glbctx_feat,
            relayed_feat=relayed_feat)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results = loss_mask
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # semantic segmentation branch
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        # global context branch
        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
            loss_glbctx = self.glbctx_head.loss(mc_pred, gt_labels)
            losses['loss_glbctx'] = loss_glbctx
        else:
            glbctx_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat, glbctx_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # refine boxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        if self.with_feat_relay:
            relayed_feat = self._slice_pos_feats(bbox_results['relayed_feat'],
                                                 sampling_results)
            relayed_feat = self.feat_relay_head(relayed_feat)
        else:
            relayed_feat = None

        mask_results = self._mask_forward_train(x, sampling_results, gt_masks,
                                                rcnn_train_cfg, semantic_feat,
                                                glbctx_feat, relayed_feat)
        mask_lw = sum(self.stage_loss_weights)
        losses['loss_mask'] = mask_lw * mask_results['loss_mask']

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
        else:
            glbctx_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head.num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i,
                x,
                rois,
                semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refine_rois = bbox_head.regress_by_class(
                            rois[j], bbox_label[j], bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refine_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        det_bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head.num_classes
                det_segm_results = [[[] for _ in range(mask_classes)]
                                    for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)

                # get relay feature on mask_rois
                bbox_results = self._bbox_forward(
                    -1,
                    x,
                    mask_rois,
                    semantic_feat=semantic_feat,
                    glbctx_feat=glbctx_feat)
                relayed_feat = bbox_results['relayed_feat']
                relayed_feat = self.feat_relay_head(relayed_feat)

                mask_results = self._mask_forward(
                    x,
                    mask_rois,
                    semantic_feat=semantic_feat,
                    glbctx_feat=glbctx_feat,
                    relayed_feat=relayed_feat)
                mask_pred = mask_results['mask_pred']

                # split batch mask prediction back to each image
                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                mask_preds = mask_pred.split(num_bbox_per_img, 0)

                # apply mask post-processing to each image individually
                det_segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        det_segm_results.append(
                            [[] for _ in range(self.mask_head.num_classes)])
                    else:
                        segm_result = self.mask_head.get_seg_masks(
                            mask_preds[i], _bboxes[i], det_labels[i],
                            self.test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        det_segm_results.append(segm_result)

        # return results
        if self.with_mask:
            return list(zip(det_bbox_results, det_segm_results))
        else:
            return det_bbox_results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1] for feat in img_feats
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        if self.with_glbctx:
            glbctx_feats = [self.glbctx_head(feat)[1] for feat in img_feats]
        else:
            glbctx_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic_feat, glbctx_feat in zip(
                img_feats, img_metas, semantic_feats, glbctx_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i,
                    x,
                    rois,
                    semantic_feat=semantic_feat,
                    glbctx_feat=glbctx_feat)
                ms_scores.append(bbox_results['cls_score'])
                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
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

        det_bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                det_segm_results = [[]
                                    for _ in range(self.mask_head.num_classes)]
            else:
                aug_masks = []
                for x, img_meta, semantic_feat, glbctx_feat in zip(
                        img_feats, img_metas, semantic_feats, glbctx_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    # get relay feature on mask_rois
                    bbox_results = self._bbox_forward(
                        -1,
                        x,
                        mask_rois,
                        semantic_feat=semantic_feat,
                        glbctx_feat=glbctx_feat)
                    relayed_feat = bbox_results['relayed_feat']
                    relayed_feat = self.feat_relay_head(relayed_feat)
                    mask_results = self._mask_forward(
                        x,
                        mask_rois,
                        semantic_feat=semantic_feat,
                        glbctx_feat=glbctx_feat,
                        relayed_feat=relayed_feat)
                    mask_pred = mask_results['mask_pred']
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks, img_metas,
                                               self.test_cfg)
                ori_shape = img_metas[0][0]['ori_shape']
                det_segm_results = self.mask_head.get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return [(det_bbox_results, det_segm_results)]
        else:
            return [det_bbox_results]

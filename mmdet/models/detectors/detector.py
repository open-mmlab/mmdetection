import torch
import torch.nn as nn

from .. import builder
from mmdet.core import (bbox2roi, bbox_mapping, split_combined_gt_polys,
                        bbox2result, multiclass_nms, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, sample_proposals)


class Detector(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_block=None,
                 bbox_head=None,
                 mask_block=None,
                 mask_head=None,
                 rpn_train_cfg=None,
                 rpn_test_cfg=None,
                 rcnn_train_cfg=None,
                 rcnn_test_cfg=None,
                 pretrained=None):
        super(Detector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        self.with_neck = True if neck is not None else False
        if self.with_neck:
            self.neck = builder.build_neck(neck)

        self.with_rpn = True if rpn_head is not None else False
        if self.with_rpn:
            self.rpn_head = builder.build_rpn_head(rpn_head)
            self.rpn_train_cfg = rpn_train_cfg
            self.rpn_test_cfg = rpn_test_cfg

        self.with_bbox = True if bbox_head is not None else False
        if self.with_bbox:
            self.bbox_roi_extractor = builder.build_roi_extractor(roi_block)
            self.bbox_head = builder.build_bbox_head(bbox_head)
            self.rcnn_train_cfg = rcnn_train_cfg
            self.rcnn_test_cfg = rcnn_test_cfg

        self.with_mask = True if mask_head is not None else False
        if self.with_mask:
            self.mask_roi_extractor = builder.build_roi_extractor(mask_block)
            self.mask_head = builder.build_mask_head(mask_head)

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
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def forward(self,
                img,
                img_meta,
                gt_bboxes=None,
                proposals=None,
                gt_labels=None,
                gt_bboxes_ignore=None,
                gt_mask_polys=None,
                gt_poly_lens=None,
                num_polys_per_mask=None,
                return_loss=True,
                return_bboxes=True,
                rescale=False):
        assert proposals is not None or self.with_rpn, "Only one of proposals file and RPN can exist."

        if not return_loss:
            return self.test(img, img_meta, proposals, rescale)
        else:
            losses = dict()

        img_shapes = img_meta['img_shape']
        x = self.backbone(img)

        if self.with_neck:
            x = self.neck(x)

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_shapes,
                                          self.rpn_train_cfg)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

        if self.with_bbox:
            if self.with_rpn:
                proposal_inputs = rpn_outs + (img_shapes, self.rpn_test_cfg)
                proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
            else:
                proposal_list = proposals

            (pos_inds, neg_inds, pos_proposals, neg_proposals,
             pos_assigned_gt_inds,
             pos_gt_bboxes, pos_gt_labels) = sample_proposals(
                 proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels,
                 self.rcnn_train_cfg)

            labels, label_weights, bbox_targets, bbox_weights = \
                self.bbox_head.get_bbox_target(
                    pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
                    self.rcnn_train_cfg)

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
            gt_polys = split_combined_gt_polys(gt_mask_polys, gt_poly_lens,
                                               num_polys_per_mask)
            mask_targets = self.mask_head.get_mask_target(
                pos_proposals, pos_assigned_gt_inds, gt_polys, img_meta,
                self.rcnn_train_cfg)
            pos_rois = bbox2roi(pos_proposals)
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)
            losses['loss_mask'] = self.mask_head.loss(mask_pred, mask_targets,
                                                      torch.cat(pos_gt_labels))
        return losses

    def test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test w/ or w/o augmentations."""
        assert isinstance(imgs, list) and isinstance(img_metas, list)
        assert len(imgs) == len(img_metas)
        img_per_gpu = imgs[0].size(0)
        assert img_per_gpu == 1
        if len(imgs) == 1:
            return self.simple_test(imgs[0], img_metas[0], proposals, rescale)
        else:
            return self.aug_test(imgs, img_metas, proposals, rescale)

    def simple_test_rpn(self, x, img_meta):
        img_shapes = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_shapes, self.rpn_test_cfg)
        proposal_list = self.rpn_head.get_proposals(*proposal_inputs)[0]
        return proposal_list

    def simple_test_bboxes(self, x, img_meta, proposals, rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        # image shape of the first image in the batch (only one)
        img_shape = img_meta['img_shape'][0]
        scale_factor = img_meta['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            nms_cfg=self.rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        img_shape = img_meta['img_shape'][0]
        scale_factor = img_meta['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor.float()
                       if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred,
                det_bboxes,
                det_labels,
                self.rcnn_test_cfg,
                ori_scale=img_meta['ori_shape'])
        return segm_result

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        # get feature maps
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if self.with_rpn:
            proposals = self.simple_test_rpn(x, img_meta)
        if self.with_bbox:
            # BUG proposals shape?
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, [proposals], rescale=rescale)
            bbox_result = bbox2result(det_bboxes, det_labels,
                                      self.bbox_head.num_classes)
            if not self.with_mask:
                return bbox_result

            segm_result = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_result, segm_result
        else:
            proposals[:, :4] /= img_meta['scale_factor'].float()
            return proposals.cpu().numpy()

    # TODO aug test haven't been verified
    def aug_test_bboxes(self, imgs, img_metas):
        """Test with augmentations for det bboxes."""
        # step 1: get RPN proposals for augmented images, apply NMS to the
        # union of all proposals.
        aug_proposals = []
        for img, img_meta in zip(imgs, img_metas):
            x = self.backbone(img)
            if self.neck is not None:
                x = self.neck(x)
            rpn_outs = self.rpn_head(x)
            proposal_inputs = rpn_outs + (img_meta['shape_scale'],
                                          self.rpn_test_cfg)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
            assert len(proposal_list) == 1
            aug_proposals.append(proposal_list[0])  # len(proposal_list) = 1
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = merge_aug_proposals(aug_proposals, img_metas,
                                               self.rpn_test_cfg)
        # step 2: Given merged proposals, predict bboxes for augmented images,
        # output the union of these bboxes.
        aug_bboxes = []
        aug_scores = []
        for img, img_meta in zip(imgs, img_metas):
            # only one image in the batch
            img_shape = img_meta['shape_scale'][0]
            flip = img_meta['flip'][0]
            proposals = bbox_mapping(merged_proposals[:, :4], img_shape, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            x = self.backbone(img)
            if self.neck is not None:
                x = self.neck(x)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                rescale=False,
                nms_cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, self.rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, self.rcnn_test_cfg.score_thr,
            self.rcnn_test_cfg.nms_thr, self.rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels

    def aug_test_mask(self,
                      imgs,
                      img_metas,
                      det_bboxes,
                      det_labels,
                      rescale=False):
        # step 3: Given merged bboxes, predict masks for augmented images,
        # scores of masks are averaged across augmented images.
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0]['shape_scale'][0][-1]
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for img, img_meta in zip(imgs, img_metas):
                img_shape = img_meta['shape_scale'][0]
                flip = img_meta['flip'][0]
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape, flip)
                mask_rois = bbox2roi([_bboxes])
                x = self.backbone(img)
                if self.neck is not None:
                    x = self.neck(x)
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.rcnn_test_cfg)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks, _det_bboxes, det_labels,
                img_metas[0]['shape_scale'][0], self.rcnn_test_cfg, rescale)
        return segm_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        if imgs[0].
        """
        # aug test det bboxes
        det_bboxes, det_labels = self.aug_test_bboxes(imgs, img_metas)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0]['shape_scale'][0][-1]
        bbox_result = bbox2result(_det_bboxes, det_labels,
                                  self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_result
        segm_result = self.aug_test_mask(
            imgs, img_metas, det_bboxes, det_labels, rescale=rescale)
        return bbox_result, segm_result

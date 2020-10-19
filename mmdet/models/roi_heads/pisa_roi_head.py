from mmdet.core import bbox2roi
from ..builder import HEADS
from ..losses.pisa_loss import carl_loss, isr_p
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class PISARoIHead(StandardRoIHead):
    r"""The RoI head for `Prime Sample Attention in Object Detection
    <https://arxiv.org/abs/1904.04821>`_."""

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """Forward function for training.

        Args:
            x (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): List of region proposals.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (list[Tensor], optional): Specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : True segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            neg_label_weights = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # neg label weight is obtained by sampling when using ISR-N
                neg_label_weight = None
                if isinstance(sampling_result, tuple):
                    sampling_result, neg_label_weight = sampling_result
                sampling_results.append(sampling_result)
                neg_label_weights.append(neg_label_weight)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
                neg_label_weights=neg_label_weights)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            img_metas,
                            neg_label_weights=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        # neg_label_weights obtained by sampler is image-wise, mapping back to
        # the corresponding location in label weights
        if neg_label_weights[0] is not None:
            label_weights = bbox_targets[1]
            cur_num_rois = 0
            for i in range(len(sampling_results)):
                num_pos = sampling_results[i].pos_inds.size(0)
                num_neg = sampling_results[i].neg_inds.size(0)
                label_weights[cur_num_rois + num_pos:cur_num_rois + num_pos +
                              num_neg] = neg_label_weights[i]
                cur_num_rois += num_pos + num_neg

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Apply ISR-P
        isr_cfg = self.train_cfg.get('isr', None)
        if isr_cfg is not None:
            bbox_targets = isr_p(
                cls_score,
                bbox_pred,
                bbox_targets,
                rois,
                sampling_results,
                self.bbox_head.loss_cls,
                self.bbox_head.bbox_coder,
                **isr_cfg,
                num_class=self.bbox_head.num_classes)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois,
                                        *bbox_targets)

        # Add CARL Loss
        carl_cfg = self.train_cfg.get('carl', None)
        if carl_cfg is not None:
            loss_carl = carl_loss(
                cls_score,
                bbox_targets[0],
                bbox_pred,
                bbox_targets[2],
                self.bbox_head.loss_bbox,
                **carl_cfg,
                num_class=self.bbox_head.num_classes)
            loss_bbox.update(loss_carl)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

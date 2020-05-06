import torch

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class MaskScoringRoIHead(StandardRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(MaskScoringRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = build_head(mask_iou_head)

    def init_weights(self, pretrained):
        super(MaskScoringRoIHead, self).init_weights(pretrained)
        self.mask_iou_head.init_weights()

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_results = super(MaskScoringRoIHead,
                             self)._mask_forward_train(x, sampling_results,
                                                       bbox_feats, gt_masks,
                                                       img_metas)
        if mask_results['loss_mask'] is None:
            return mask_results

        # mask iou head forward and loss
        pos_mask_pred = mask_results['mask_pred'][
            range(mask_results['mask_pred'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                           pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                          pos_labels]

        mask_iou_targets = self.mask_iou_head.get_targets(
            sampling_results, gt_masks, pos_mask_pred,
            mask_results['mask_targets'], self.train_cfg)
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                mask_iou_targets)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            mask_scores = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] *
                det_bboxes.new_tensor(scale_factor) if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
            # get mask scores with mask iou head
            mask_iou_pred = self.mask_iou_head(
                mask_results['mask_feats'],
                mask_results['mask_pred'][range(det_labels.size(0)),
                                          det_labels])
            mask_scores = self.mask_iou_head.get_mask_scores(
                mask_iou_pred, det_bboxes, det_labels)
        return segm_result, mask_scores

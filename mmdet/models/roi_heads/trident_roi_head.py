import torch

from mmdet.core import bbox2result
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from ...ops import batched_nms
from ..builder import HEADS

def merge_branch(det_bboxes, det_labels, test_cfg):
    pass


@HEADS.register_module()
class TridentRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head.
    """

    def __init__(self, num_branch, test_branch_idx, **kwargs):
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super(TridentRoIHead, self).__init__(**kwargs)

    # def simple_test(self,
    #                 x,
    #                 proposal_list,
    #                 img_metas,
    #                 proposals=None,
    #                 rescale=False):
    #     """Test without augmentation."""
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #
    #     assert len(img_metas) == 1
    #     num_branch = (self.num_branch
    #                   if self.test_branch_idx == -1 else 1)
    #
    #     det_bboxes_list, det_labels_list = [], []
    #     for _ in range(num_branch):
    #         det_bboxes, det_labels = self.simple_test_bboxes(
    #             x,
    #             img_metas,
    #             proposal_list[_:_ + 1],
    #             self.test_cfg,
    #             rescale=rescale)
    #         det_bboxes_list.append(det_bboxes)
    #         det_labels_list.append(det_labels)
    #
    #     trident_det_bboxes = torch.cat(det_bboxes_list, 0)
    #     trident_det_labels = torch.cat(det_labels_list, 0)
    #     # bbox_results = bbox2result(trident_det_bboxes, trident_det_labels,
    #     #                            self.bbox_head.num_classes)
    #     # return bbox_results
    #
    #     if trident_det_bboxes.numel() == 0:
    #         det_bboxes = trident_det_bboxes.new_zeros((0, 5))
    #         det_labels = trident_det_bboxes.new_zeros((0,), dtype=torch.long)
    #     else:
    #         nms_bboxes = trident_det_bboxes[:, :4]
    #         nms_scores = trident_det_bboxes[:, 4]
    #         nms_inds = trident_det_labels
    #         nms_cfg = self.test_cfg['nms']
    #         det_bboxes, keep = batched_nms(nms_bboxes, nms_scores,
    #                                        nms_inds, nms_cfg)
    #         det_labels = trident_det_labels[keep]
    #         if self.test_cfg['max_per_img'] > 0:
    #             det_labels = det_labels[:self.test_cfg['max_per_img']]
    #             det_bboxes = det_bboxes[:self.test_cfg['max_per_img']]
    #
    #     bbox_results = bbox2result(det_bboxes, det_labels,
    #                                self.bbox_head.num_classes)
    #
    #     if not self.with_mask:
    #         return bbox_results
    #     else:
    #         segm_results = self.simple_test_mask(
    #             x, img_metas, det_bboxes, det_labels, rescale=rescale)
    #         return bbox_results, segm_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_weights=None):

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
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
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

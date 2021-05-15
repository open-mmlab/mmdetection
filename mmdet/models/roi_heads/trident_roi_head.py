import torch
from mmcv.ops import batched_nms

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        multiclass_nms)
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from ..builder import HEADS


@HEADS.register_module()
class TridentRoIHead(StandardRoIHead):
    """Trident roi head.

    Args:
        num_branch (int): Number of branches in TridentNet.
        test_branch_idx (int): In inference, all 3 branches will be used
            if `test_branch_idx==-1`, otherwise only branch with index
            `test_branch_idx` will be used.
    """

    def __init__(self, num_branch, test_branch_idx, **kwargs):
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super(TridentRoIHead, self).__init__(**kwargs)

    def merge_trident_bboxes(self, trident_det_bboxes, trident_det_labels):
        """Merge bbox predictions of each branch."""
        if trident_det_bboxes.numel() == 0:
            det_bboxes = trident_det_bboxes.new_zeros((0, 5))
            det_labels = trident_det_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            nms_bboxes = trident_det_bboxes[:, :4]
            nms_scores = trident_det_bboxes[:, 4].contiguous()
            nms_inds = trident_det_labels
            nms_cfg = self.test_cfg['nms']
            det_bboxes, keep = batched_nms(nms_bboxes, nms_scores, nms_inds,
                                           nms_cfg)
            det_labels = trident_det_labels[keep]
            if self.test_cfg['max_per_img'] > 0:
                det_labels = det_labels[:self.test_cfg['max_per_img']]
                det_bboxes = det_bboxes[:self.test_cfg['max_per_img']]

        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation as follows:

        1. Compute prediction bbox and label per branch.
        2. Merge predictions of each branch according to scores of
           bboxes, i.e., bboxes with higher score are kept to give
           top-k prediction.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes_list, det_labels_list = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        num_branch = self.num_branch if self.test_branch_idx == -1 else 1
        for _ in range(len(det_bboxes_list)):
            if det_bboxes_list[_].shape[0] == 0:
                det_bboxes_list[_] = det_bboxes_list[_].new_empty((0, 5))
        det_bboxes, det_labels = [], []
        for i in range(len(img_metas) // num_branch):
            det_result = self.merge_trident_bboxes(
                torch.cat(det_bboxes_list[i * num_branch:(i + 1) *
                                          num_branch]),
                torch.cat(det_labels_list[i * num_branch:(i + 1) *
                                          num_branch]))
            det_bboxes.append(det_result[0])
            det_labels.append(det_result[1])

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        return bbox_results

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

            trident_bboxes, trident_scores = [], []
            for branch_idx in range(len(proposal_list)):
                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                         scale_factor, flip, flip_direction)
                rois = bbox2roi([proposals])
                bbox_results = self._bbox_forward(x, rois)
                bboxes, scores = self.bbox_head.get_bboxes(
                    rois,
                    bbox_results['cls_score'],
                    bbox_results['bbox_pred'],
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None)
                trident_bboxes.append(bboxes)
                trident_scores.append(scores)

            aug_bboxes.append(torch.cat(trident_bboxes, 0))
            aug_scores.append(torch.cat(trident_scores, 0))
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels

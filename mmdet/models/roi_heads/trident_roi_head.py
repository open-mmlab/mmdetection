import torch
from mmcv.ops import batched_nms

from mmdet.core import bbox2result
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
            x,
            img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale)

        trident_det_bboxes = torch.cat(det_bboxes_list, 0)
        trident_det_labels = torch.cat(det_labels_list, 0)

        if trident_det_bboxes.numel() == 0:
            det_bboxes = trident_det_bboxes.new_zeros((0, 5))
            det_labels = trident_det_bboxes.new_zeros((0,), dtype=torch.long)
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

        det_bboxes, det_labels = [det_bboxes], [det_labels]

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

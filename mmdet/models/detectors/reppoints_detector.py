import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class RepPointsDetector(SingleStageDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results

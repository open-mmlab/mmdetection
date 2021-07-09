import torch

from mmdet.models.builder import HEADS
from .base_pan_head import BasePanHead


@HEADS.register_module()
class HeuristicPanHead(BasePanHead):
    """Base class for panoptic heads."""

    def __init__(self, num_things=80, num_stuff=53, init_cfg=None, **kwargs):
        super(HeuristicPanHead, self).__init__(num_things, num_stuff, None,
                                               init_cfg, **kwargs)

    def forward_train(self, gt_masks=None, gt_semantic_seg=None, **kwargs):
        """HeuristicPanHead has not training loss."""
        return dict()

    def _lay_masks(self, bboxes, labels, masks, img_shape, overlap_thr=0.5):
        num_insts = bboxes.shape[0]
        id_map = torch.zeros(img_shape, device=labels.device, dtype=torch.long)
        if num_insts == 0:
            return id_map, labels

        scores, bboxes = bboxes[:, -1], bboxes[:, :4]

        # for unmatched shits, order by scores
        order = torch.argsort(-scores)
        bboxes = bboxes[order]
        labels = labels[order]
        segm_masks = masks[order]

        inst_idx = 1
        left_labels = []
        for idx in range(bboxes.shape[0]):
            _cls = labels[idx]
            _mask = segm_masks[idx]
            inst_id_map = torch.ones_like(_mask, dtype=torch.long) * inst_idx
            area = _mask.sum()
            if area == 0:
                continue

            used = id_map > 0
            intersect = (_mask * used).sum()
            if (intersect / (area + 1e-5)) > overlap_thr:
                continue

            _part = _mask * (~used)
            id_map = torch.where(_part, inst_id_map, id_map)
            left_labels.append(_cls)
            inst_idx += 1

        inst_labels = torch.stack(left_labels)
        assert inst_idx == (len(inst_labels) + 1)
        return id_map, inst_labels

    def simple_test(self, img_metas, det_bboxes, det_labels, mask_preds,
                    seg_logits, **kwargs):
        """Test without augmentation."""

import torch

from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class HeuristicPanHead(BasePanopticFusionHead):
    """Fusion Head with Heuristic method."""

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(HeuristicPanHead,
              self).__init__(num_things_classes, num_stuff_classes, test_cfg,
                             None, init_cfg, **kwargs)

    def forward_train(self, gt_masks=None, gt_semantic_seg=None, **kwargs):
        """HeuristicPanHead has not training loss."""
        return dict()

    def _lay_masks(self, bboxes, labels, masks, overlap_thr=0.5):
        num_insts = bboxes.shape[0]
        id_map = torch.zeros(
            masks.shape[-2:], device=bboxes.device, dtype=torch.long)
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

        if len(left_labels) > 0:
            inst_labels = torch.stack(left_labels)
        else:
            inst_labels = bboxes.new_zeros((0, ), dtype=torch.long)
        assert inst_idx == (len(inst_labels) + 1)
        return id_map, inst_labels

    def simple_test(self, det_bboxes, det_labels, mask_preds, seg_logits,
                    **kwargs):
        mask_preds = mask_preds >= self.test_cfg.mask_thr_binary
        id_map, labels = self._lay_masks(det_bboxes, det_labels, mask_preds,
                                         self.test_cfg.mask_overlap)

        seg_pred = seg_logits.argmax(dim=0)
        seg_pred = seg_pred + self.num_things_classes

        pano_results = seg_pred
        instance_id = 1
        for idx in range(det_labels.shape[0]):
            _mask = id_map == (idx + 1)
            if _mask.sum() == 0:
                continue
            _cls = labels[idx]
            # simply trust detection
            segment_id = _cls + instance_id * INSTANCE_OFFSET
            pano_results[_mask] = segment_id
            instance_id += 1

        ids, cnts = torch.unique(
            pano_results % INSTANCE_OFFSET, return_counts=True)
        stuff_ids = ids[ids >= self.num_things_classes]
        stuff_cnts = cnts[ids >= self.num_things_classes]
        ignore_stuff_ids = stuff_ids[
            stuff_cnts < self.test_cfg.stuff_area_limit]

        assert pano_results.ndim == 2
        pano_results[(pano_results.unsqueeze(2) == ignore_stuff_ids.reshape(
            1, 1, -1)).any(dim=2)] = self.num_classes

        return pano_results

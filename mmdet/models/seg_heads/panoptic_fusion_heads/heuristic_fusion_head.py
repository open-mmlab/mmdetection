# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.data import PixelData
from torch import Tensor

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.utils import OptConfigType, OptMultiConfig, PixelList
from mmdet.registry import MODELS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@MODELS.register_module()
class HeuristicFusionHead(BasePanopticFusionHead):
    """Fusion Head with Heuristic method."""

    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            test_cfg=test_cfg,
            loss_panoptic=None,
            init_cfg=init_cfg,
            **kwargs)

    def loss(self, **kwargs) -> dict:
        """HeuristicFusionHead has no training loss."""
        return dict()

    def _lay_masks(self,
                   bboxes: Tensor,
                   labels: Tensor,
                   masks: Tensor,
                   overlap_thr: float = 0.5) -> Tensor:
        """Lay instance masks to a result map.

        Args:
            bboxes (Tensor): The bboxes results, (K, 4).
            labels (Tensor): The labels of bboxes, (K, ).
            masks (Tensor): The instance masks, (K, H, W).
            overlap_thr (float): Threshold to determine whether two masks
                overlap. default: 0.5.

        Returns:
            Tensor: The result map, (H, W).
        """
        num_insts = bboxes.shape[0]
        id_map = torch.zeros(
            masks.shape[-2:], device=bboxes.device, dtype=torch.long)
        if num_insts == 0:
            return id_map, labels

        scores, bboxes = bboxes[:, -1], bboxes[:, :4]

        # Sort by score to use heuristic fusion
        order = torch.argsort(-scores)
        bboxes = bboxes[order]
        labels = labels[order]
        segm_masks = masks[order]

        instance_id = 1
        left_labels = []
        for idx in range(bboxes.shape[0]):
            _cls = labels[idx]
            _mask = segm_masks[idx]
            instance_id_map = torch.ones_like(
                _mask, dtype=torch.long) * instance_id
            area = _mask.sum()
            if area == 0:
                continue

            pasted = id_map > 0
            intersect = (_mask * pasted).sum()
            if (intersect / (area + 1e-5)) > overlap_thr:
                continue

            _part = _mask * (~pasted)
            id_map = torch.where(_part, instance_id_map, id_map)
            left_labels.append(_cls)
            instance_id += 1

        if len(left_labels) > 0:
            instance_labels = torch.stack(left_labels)
        else:
            instance_labels = bboxes.new_zeros((0, ), dtype=torch.long)
        assert instance_id == (len(instance_labels) + 1)
        return id_map, instance_labels

    def _predict_single(self, det_bboxes: Tensor, det_labels: Tensor,
                        mask_preds: Tensor, seg_preds: Tensor,
                        **kwargs) -> PixelData:
        """Fuse the results of instance and semantic segmentations.

        Args:
            det_bboxes (Tensor): The bboxes results, (K, 4).
            det_labels (Tensor): The labels of bboxes, (K,).
            mask_preds (Tensor): The masks results, (K, H, W).
            seg_preds (Tensor): The semantic segmentation results,
                (num_stuff + 1, H, W).

        Returns:
            Tensor: The panoptic segmentation result, (H, W).
        """
        mask_preds = mask_preds >= self.test_cfg.mask_thr_binary
        id_map, labels = self._lay_masks(det_bboxes, det_labels, mask_preds,
                                         self.test_cfg.mask_overlap)

        seg_results = seg_preds.argmax(dim=0)
        seg_results = seg_results + self.num_things_classes

        pan_results = seg_results
        instance_id = 1
        for idx in range(det_labels.shape[0]):
            _mask = id_map == (idx + 1)
            if _mask.sum() == 0:
                continue
            _cls = labels[idx]
            # simply trust detection
            segment_id = _cls + instance_id * INSTANCE_OFFSET
            pan_results[_mask] = segment_id
            instance_id += 1

        ids, counts = torch.unique(
            pan_results % INSTANCE_OFFSET, return_counts=True)
        stuff_ids = ids[ids >= self.num_things_classes]
        stuff_counts = counts[ids >= self.num_things_classes]
        ignore_stuff_ids = stuff_ids[
            stuff_counts < self.test_cfg.stuff_area_limit]

        assert pan_results.ndim == 2
        pan_results[(pan_results.unsqueeze(2) == ignore_stuff_ids.reshape(
            1, 1, -1)).any(dim=2)] = self.num_classes

        pan_results = PixelData(sem_seg=pan_results[None].int())
        return pan_results

    def predict(self, det_bboxes_list: List[Tensor],
                det_labels_list: List[Tensor], mask_preds_list: List[Tensor],
                seg_preds_list: Tensor, **kwargs) -> PixelList:
        """Predict results by fusing the results of instance and semantic
        segmentations.

        Args:
            det_bboxes_list (List[Tensor]): List of bboxes results.
            det_labels_list (List[Tensor]): List of labels of bboxes.
            mask_preds_list (List[Tensor]): List of masks results.
            seg_preds_list (Tensor): List of semantic segmentation results.

        Returns:
            List[PixelData]: Panoptic segmentation result.
        """
        results_list = [
            self._predict_single(det_bboxes_list[i], det_labels_list[i],
                                 mask_preds_list[i], seg_preds_list[i])
            for i in range(len(det_bboxes_list))
        ]

        return results_list

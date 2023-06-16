# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class RefSegMetric(BaseMetric):
    """Referring Expression Segmentation Metric."""

    def __init__(self, metric: Sequence = ('cIoU', 'mIoU'), **kwargs):
        super().__init__(**kwargs)
        assert set(metric).issubset(['cIoU', 'mIoU']), \
            f'Only support cIoU and mIoU, but got {metric}'
        assert len(metric) > 0, 'metrics should not be empty'
        self.metrics = metric

    def compute_iou(self, pred_seg: torch.Tensor,
                    gt_seg: torch.Tensor) -> tuple:
        overlap = pred_seg & gt_seg
        union = pred_seg | gt_seg
        return overlap, union

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_instances']['masks'].bool()
            label = data_sample['gt_masks'].to_tensor(
                pred_label.dtype, pred_label.device).bool()
            # calculate iou
            overlap, union = self.compute_iou(pred_label, label)

            bs = len(pred_label)
            iou = overlap.reshape(bs, -1).sum(-1) * 1.0 / union.reshape(
                bs, -1).sum(-1)
            iou = torch.nan_to_num_(iou, nan=0.0)
            self.results.append((overlap.sum(), union.sum(), iou.sum(), bs))

    def compute_metrics(self, results: list) -> dict:
        results = tuple(zip(*results))
        assert len(results) == 4
        cum_i = sum(results[0])
        cum_u = sum(results[1])
        iou = sum(results[2])
        seg_total = sum(results[3])

        metrics = {}
        if 'cIoU' in self.metrics:
            metrics['cIoU'] = cum_i * 100 / cum_u
        if 'mIoU' in self.metrics:
            metrics['mIoU'] = iou * 100 / seg_total
        return metrics

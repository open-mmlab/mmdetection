# Copyright (c) OpenMMLab. All rights reserved
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from ..functional import bbox_overlaps


class RecallTracker:
    """Utility class to track recall@k for various k, split by categories."""

    def __init__(self, topk: Sequence[int]):
        """
        Parameters:
           - topk : tuple of ints corresponding to the recalls being
           tracked (eg, recall@1, recall@10, ...)
        """

        self.total_byk_bycat: Dict[int, Dict[str, int]] = {
            k: defaultdict(int)
            for k in topk
        }
        self.positives_byk_bycat: Dict[int, Dict[str, int]] = {
            k: defaultdict(int)
            for k in topk
        }

    def add_positive(self, k: int, category: str):
        """Log a positive hit @k for given category."""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f'{k} is not a valid recall threshold')
        self.total_byk_bycat[k][category] += 1
        self.positives_byk_bycat[k][category] += 1

    def add_negative(self, k: int, category: str):
        """Log a negative hit @k for given category."""
        if k not in self.total_byk_bycat:
            raise RuntimeError(f'{k} is not a valid recall threshold')
        self.total_byk_bycat[k][category] += 1

    def report(self) -> Dict[str, Dict[str, float]]:
        """Return a condensed report of the results as a dict of dict.

        report[k][cat] is the recall@k for the given category
        """
        report: Dict[str, Dict[str, float]] = {}
        for k in self.total_byk_bycat:
            assert k in self.positives_byk_bycat
            report[str(k)] = {
                cat:
                self.positives_byk_bycat[k][cat] / self.total_byk_bycat[k][cat]
                for cat in self.total_byk_bycat[k]
            }
        return report


@METRICS.register_module()
class Flickr30kMetric(BaseMetric):
    """Phrase Grounding Metric."""

    def __init__(
        self,
        topk: Sequence[int] = (1, 5, 10, -1),
        iou_thrs: float = 0.5,
        merge_boxes: bool = False,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.iou_thrs = iou_thrs
        self.topk = topk
        self.merge = merge_boxes

    def merge_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """Return the boxes corresponding to the smallest enclosing box
        containing all the provided boxes The boxes are expected in [x1, y1,
        x2, y2] format."""
        if len(boxes) == 1:
            return boxes

        np_boxes = np.asarray(boxes)

        return [[
            np.boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(),
            np_boxes[:, 3].max()
        ]]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']['bboxes']
            gt_label = data_sample['phrase_ids']
            phrases = data_sample['phrases']
            assert len(gt) == len(gt_label)

            self.results.append((pred, gt, gt_label, phrases))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        pred_list, gt_list, gt_label_list, phrase_list = zip(*results)

        recall_tracker = RecallTracker(self.topk)

        for pred, gt_boxes, gt_labels, phrases in zip(pred_list, gt_list,
                                                      gt_label_list,
                                                      phrase_list):
            pred_boxes = pred['bboxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            for i, phrase in enumerate(phrases):
                cur_index = pred_labels == i
                cur_boxes = pred_boxes[cur_index]
                tar_index = [
                    index for index, value in enumerate(gt_labels)
                    if value == i
                ]
                tar_boxes = gt_boxes[tar_index]
                if self.merge:
                    tar_boxes = self.merge_boxes(tar_boxes)
                if len(cur_boxes) == 0:
                    cur_boxes = [[0., 0., 0., 0.]]
                ious = bbox_overlaps(
                    np.asarray(cur_boxes), np.asarray(tar_boxes))
                for k in self.topk:
                    if k == -1:
                        maxi = ious.max()
                    else:
                        assert k > 0
                        maxi = ious[:k].max()
                    if maxi >= self.iou_thrs:
                        recall_tracker.add_positive(k, 'all')
                        # TODO: do not support class-wise evaluation yet
                        # for phrase_type in phrase['phrase_type']:
                        #     recall_tracker.add_positive(k, phrase_type)
                    else:
                        recall_tracker.add_negative(k, 'all')
                        # for phrase_type in phrase['phrase_type']:
                        #     recall_tracker.add_negative(k, phrase_type)

        results = recall_tracker.report()
        logger.info(results)
        return results

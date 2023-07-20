# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmengine
import numpy as np
from mmengine.fileio import load
from mmengine.logging import print_log
from pycocotools import mask as coco_mask
from terminaltables import AsciiTable

from mmdet.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class CocoOccludedSeparatedMetric(CocoMetric):
    """Metric of separated and occluded masks which presented in paper `A Tri-
    Layer Plugin to Improve Occluded Detection.

    <https://arxiv.org/abs/2210.10046>`_.

    Separated COCO and Occluded COCO are automatically generated subsets of
    COCO val dataset, collecting separated objects and partially occluded
    objects for a large variety of categories. In this way, we define
    occlusion into two major categories: separated and partially occluded.

    - Separation: target object segmentation mask is separated into distinct
      regions by the occluder.
    - Partial Occlusion: target object is partially occluded but the
      segmentation mask is connected.

    These two new scalable real-image datasets are to benchmark a model's
    capability to detect occluded objects of 80 common categories.

    Please cite the paper if you use this dataset:

    @article{zhan2022triocc,
        title={A Tri-Layer Plugin to Improve Occluded Detection},
        author={Zhan, Guanqi and Xie, Weidi and Zisserman, Andrew},
        journal={British Machine Vision Conference},
        year={2022}
    }

    Args:
        occluded_ann (str): Path to the occluded coco annotation file.
        separated_ann (str): Path to the separated coco annotation file.
        score_thr (float): Score threshold of the detection masks.
            Defaults to 0.3.
        iou_thr (float): IoU threshold for the recall calculation.
            Defaults to 0.75.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(
            self,
            *args,
            occluded_ann:
        str = 'https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/occluded_coco.pkl',  # noqa
            separated_ann:
        str = 'https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/separated_coco.pkl',  # noqa
            score_thr: float = 0.3,
            iou_thr: float = 0.75,
            metric: Union[str, List[str]] = ['bbox', 'segm'],
            **kwargs) -> None:
        super().__init__(*args, metric=metric, **kwargs)
        self.occluded_ann = load(occluded_ann)
        self.separated_ann = load(separated_ann)
        self.score_thr = score_thr
        self.iou_thr = iou_thr

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        coco_metric_res = super().compute_metrics(results)
        eval_res = self.evaluate_occluded_separated(results)
        coco_metric_res.update(eval_res)
        return coco_metric_res

    def evaluate_occluded_separated(self, results: List[tuple]) -> dict:
        """Compute the recall of occluded and separated masks.

        Args:
            results (list[tuple]): Testing results of the dataset.

        Returns:
            dict[str, float]: The recall of occluded and separated masks.
        """
        dict_det = {}
        print_log('processing detection results...')
        prog_bar = mmengine.ProgressBar(len(results))
        for i in range(len(results)):
            gt, dt = results[i]
            img_id = dt['img_id']
            cur_img_name = self._coco_api.imgs[img_id]['file_name']
            if cur_img_name not in dict_det.keys():
                dict_det[cur_img_name] = []

            for bbox, score, label, mask in zip(dt['bboxes'], dt['scores'],
                                                dt['labels'], dt['masks']):
                cur_binary_mask = coco_mask.decode(mask)
                dict_det[cur_img_name].append([
                    score, self.dataset_meta['classes'][label],
                    cur_binary_mask, bbox
                ])
            dict_det[cur_img_name].sort(
                key=lambda x: (-x[0], x[3][0], x[3][1])
            )  # rank by confidence from high to low, avoid same confidence
            prog_bar.update()
        print_log('\ncomputing occluded mask recall...', logger='current')
        occluded_correct_num, occluded_recall = self.compute_recall(
            dict_det, gt_ann=self.occluded_ann, is_occ=True)
        print_log(
            f'\nCOCO occluded mask recall: {occluded_recall:.2f}%',
            logger='current')
        print_log(
            f'COCO occluded mask success num: {occluded_correct_num}',
            logger='current')
        print_log('computing separated mask recall...', logger='current')
        separated_correct_num, separated_recall = self.compute_recall(
            dict_det, gt_ann=self.separated_ann, is_occ=False)
        print_log(
            f'\nCOCO separated mask recall: {separated_recall:.2f}%',
            logger='current')
        print_log(
            f'COCO separated mask success num: {separated_correct_num}',
            logger='current')
        table_data = [
            ['mask type', 'recall', 'num correct'],
            ['occluded', f'{occluded_recall:.2f}%', occluded_correct_num],
            ['separated', f'{separated_recall:.2f}%', separated_correct_num]
        ]
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger='current')
        return dict(
            occluded_recall=occluded_recall, separated_recall=separated_recall)

    def compute_recall(self,
                       result_dict: dict,
                       gt_ann: list,
                       is_occ: bool = True) -> tuple:
        """Compute the recall of occluded or separated masks.

        Args:
            result_dict (dict): Processed mask results.
            gt_ann (list): Occluded or separated coco annotations.
            is_occ (bool): Whether the annotation is occluded mask.
                Defaults to True.
        Returns:
            tuple: number of correct masks and the recall.
        """
        correct = 0
        prog_bar = mmengine.ProgressBar(len(gt_ann))
        for iter_i in range(len(gt_ann)):
            cur_item = gt_ann[iter_i]
            cur_img_name = cur_item[0]
            cur_gt_bbox = cur_item[3]
            if is_occ:
                cur_gt_bbox = [
                    cur_gt_bbox[0], cur_gt_bbox[1],
                    cur_gt_bbox[0] + cur_gt_bbox[2],
                    cur_gt_bbox[1] + cur_gt_bbox[3]
                ]
            cur_gt_class = cur_item[1]
            cur_gt_mask = coco_mask.decode(cur_item[4])

            assert cur_img_name in result_dict.keys()
            cur_detections = result_dict[cur_img_name]

            correct_flag = False
            for i in range(len(cur_detections)):
                cur_det_confidence = cur_detections[i][0]
                if cur_det_confidence < self.score_thr:
                    break
                cur_det_class = cur_detections[i][1]
                if cur_det_class != cur_gt_class:
                    continue
                cur_det_mask = cur_detections[i][2]
                cur_iou = self.mask_iou(cur_det_mask, cur_gt_mask)
                if cur_iou >= self.iou_thr:
                    correct_flag = True
                    break
            if correct_flag:
                correct += 1
            prog_bar.update()
        recall = correct / len(gt_ann) * 100
        return correct, recall

    def mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Compute IoU between two masks."""
        mask1_area = np.count_nonzero(mask1 == 1)
        mask2_area = np.count_nonzero(mask2 == 1)
        intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
        iou = intersection / (mask1_area + mask2_area - intersection)
        return iou

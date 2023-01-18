# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.fileio import load
from mmcv.utils import print_log
from pycocotools import mask as coco_mask
from terminaltables import AsciiTable

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class OccludedSeparatedCocoDataset(CocoDataset):
    """COCO dataset with evaluation on separated and occluded masks which
    presented in paper `A Tri-Layer Plugin to Improve Occluded Detection.

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
    """  # noqa

    def __init__(
            self,
            *args,
            occluded_ann='https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/occluded_coco.pkl',  # noqa
            separated_ann='https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/separated_coco.pkl',  # noqa
            **kwargs):
        super().__init__(*args, **kwargs)

        # load from local file
        if osp.isfile(occluded_ann) and not osp.isabs(occluded_ann):
            occluded_ann = osp.join(self.data_root, occluded_ann)
        if osp.isfile(separated_ann) and not osp.isabs(separated_ann):
            separated_ann = osp.join(self.data_root, separated_ann)

        self.occluded_ann = load(occluded_ann)
        self.separated_ann = load(separated_ann)

    def evaluate(self,
                 results,
                 metric=[],
                 score_thr=0.3,
                 iou_thr=0.75,
                 **kwargs):
        """Occluded and separated mask evaluation in COCO protocol.

        Args:
            results (list[tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'. Defaults to [].
            score_thr (float): Score threshold of the detection masks.
                Defaults to 0.3.
            iou_thr (float): IoU threshold for the recall calculation.
                Defaults to 0.75.
        Returns:
            dict[str, float]: The recall of occluded and separated masks and
            COCO style evaluation metric.
        """
        coco_metric_res = super().evaluate(results, metric=metric, **kwargs)
        eval_res = self.evaluate_occluded_separated(results, score_thr,
                                                    iou_thr)
        coco_metric_res.update(eval_res)
        return coco_metric_res

    def evaluate_occluded_separated(self,
                                    results,
                                    score_thr=0.3,
                                    iou_thr=0.75):
        """Compute the recall of occluded and separated masks.

        Args:
            results (list[tuple]): Testing results of the dataset.
            score_thr (float): Score threshold of the detection masks.
                Defaults to 0.3.
            iou_thr (float): IoU threshold for the recall calculation.
                Defaults to 0.75.
        Returns:
            dict[str, float]: The recall of occluded and separated masks.
        """
        dict_det = {}
        print_log('processing detection results...')
        prog_bar = mmcv.ProgressBar(len(results))
        for i in range(len(results)):
            cur_img_name = self.data_infos[i]['filename']
            if cur_img_name not in dict_det.keys():
                dict_det[cur_img_name] = []
            for cat_id in range(len(results[i][1])):
                assert len(results[i][1][cat_id]) == len(results[i][0][cat_id])
                for instance_id in range(len(results[i][1][cat_id])):
                    cur_binary_mask = coco_mask.decode(
                        results[i][1][cat_id][instance_id])
                    cur_det_bbox = results[i][0][cat_id][instance_id][:4]
                    dict_det[cur_img_name].append([
                        results[i][0][cat_id][instance_id][4],
                        self.CLASSES[cat_id], cur_binary_mask, cur_det_bbox
                    ])
            dict_det[cur_img_name].sort(
                key=lambda x: (-x[0], x[3][0], x[3][1])
            )  # rank by confidence from high to low, avoid same confidence
            prog_bar.update()
        print_log('\ncomputing occluded mask recall...')
        occluded_correct_num, occluded_recall = self.compute_recall(
            dict_det,
            gt_ann=self.occluded_ann,
            score_thr=score_thr,
            iou_thr=iou_thr,
            is_occ=True)
        print_log(f'\nCOCO occluded mask recall: {occluded_recall:.2f}%')
        print_log(f'COCO occluded mask success num: {occluded_correct_num}')
        print_log('computing separated mask recall...')
        separated_correct_num, separated_recall = self.compute_recall(
            dict_det,
            gt_ann=self.separated_ann,
            score_thr=score_thr,
            iou_thr=iou_thr,
            is_occ=False)
        print_log(f'\nCOCO separated mask recall: {separated_recall:.2f}%')
        print_log(f'COCO separated mask success num: {separated_correct_num}')
        table_data = [
            ['mask type', 'recall', 'num correct'],
            ['occluded', f'{occluded_recall:.2f}%', occluded_correct_num],
            ['separated', f'{separated_recall:.2f}%', separated_correct_num]
        ]
        table = AsciiTable(table_data)
        print_log('\n' + table.table)
        return dict(
            occluded_recall=occluded_recall, separated_recall=separated_recall)

    def compute_recall(self,
                       result_dict,
                       gt_ann,
                       score_thr=0.3,
                       iou_thr=0.75,
                       is_occ=True):
        """Compute the recall of occluded or separated masks.

        Args:
            results (list[tuple]): Testing results of the dataset.
            gt_ann (list): Occluded or separated coco annotations.
            score_thr (float): Score threshold of the detection masks.
                Defaults to 0.3.
            iou_thr (float): IoU threshold for the recall calculation.
                Defaults to 0.75.
            is_occ (bool): Whether the annotation is occluded mask.
                Defaults to True.
        Returns:
            tuple: number of correct masks and the recall.
        """
        correct = 0
        prog_bar = mmcv.ProgressBar(len(gt_ann))
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
                if cur_det_confidence < score_thr:
                    break
                cur_det_class = cur_detections[i][1]
                if cur_det_class != cur_gt_class:
                    continue
                cur_det_mask = cur_detections[i][2]
                cur_iou = self.mask_iou(cur_det_mask, cur_gt_mask)
                if cur_iou >= iou_thr:
                    correct_flag = True
                    break
            if correct_flag:
                correct += 1
            prog_bar.update()
        recall = correct / len(gt_ann) * 100
        return correct, recall

    def mask_iou(self, mask1, mask2):
        """Compute IoU between two masks."""
        mask1_area = np.count_nonzero(mask1 == 1)
        mask2_area = np.count_nonzero(mask2 == 1)
        intersection = np.count_nonzero(np.logical_and(mask1 == 1, mask2 == 1))
        iou = intersection / (mask1_area + mask2_area - intersection)
        return iou

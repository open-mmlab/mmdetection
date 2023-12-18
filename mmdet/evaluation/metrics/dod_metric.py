# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS


@METRICS.register_module()
class DODCocoMetric(BaseMetric):

    default_prefix: Optional[str] = 'dod'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 collect_device: str = 'cpu',
                 outfile_prefix: Optional[str] = None,
                 backend_args: dict = None,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.outfile_prefix = outfile_prefix
        with get_local_path(ann_file, backend_args=backend_args) as local_path:
            self._coco_api = COCO(local_path)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()

            result['labels'] = pred['labels'].cpu().numpy()
            result['labels'] = data_sample['sent_ids'][result['labels']]
            self.results.append(result)

    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict]) -> list:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = label
                bbox_json_results.append(data)
        return bbox_json_results

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        result_files = self.results2json(results)
        d3_res = self._coco_api.loadRes(result_files)
        cocoEval = COCOeval(self._coco_api, d3_res, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        aps = cocoEval.eval['precision'][:, :, :, 0, -1]
        category_ids = self._coco_api.getCatIds()
        category_names = [
            cat['name'] for cat in self._coco_api.loadCats(category_ids)
        ]

        aps_lens = defaultdict(list)
        counter_lens = defaultdict(int)
        for i in range(len(category_names)):
            ap = aps[:, :, i]
            ap_value = ap[ap > -1].mean()
            if not np.isnan(ap_value):
                len_ref = len(category_names[i].split(' '))
                aps_lens[len_ref].append(ap_value)
                counter_lens[len_ref] += 1

        ap_sum_short = sum([sum(aps_lens[i]) for i in range(0, 4)])
        ap_sum_mid = sum([sum(aps_lens[i]) for i in range(4, 7)])
        ap_sum_long = sum([sum(aps_lens[i]) for i in range(7, 10)])
        ap_sum_very_long = sum([
            sum(aps_lens[i]) for i in range(10,
                                            max(counter_lens.keys()) + 1)
        ])
        c_sum_short = sum([counter_lens[i] for i in range(1, 4)])
        c_sum_mid = sum([counter_lens[i] for i in range(4, 7)])
        c_sum_long = sum([counter_lens[i] for i in range(7, 10)])
        c_sum_very_long = sum(
            [counter_lens[i] for i in range(10,
                                            max(counter_lens.keys()) + 1)])
        map_short = ap_sum_short / c_sum_short
        map_mid = ap_sum_mid / c_sum_mid
        map_long = ap_sum_long / c_sum_long
        map_very_long = ap_sum_very_long / c_sum_very_long

        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

        eval_results = {}
        for metric_item in metric_items:
            key = f'{metric_item}'
            val = cocoEval.stats[coco_metric_names[metric_item]]
            eval_results[key] = float(f'{round(val, 3)}')

        ap = cocoEval.stats[:6]
        logger.info(f'mAP_copypaste: {ap[0]:.3f} '
                    f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')

        logger.info(f'mAP over reference length: short - {map_short:.4f}, '
                    f'mid - {map_mid:.4f}, long - {map_long:.4f}, '
                    f'very long - {map_very_long:.4f}')
        eval_results['mAP_short'] = float(f'{round(map_short, 3)}')
        eval_results['mAP_mid'] = float(f'{round(map_mid, 3)}')
        eval_results['mAP_long'] = float(f'{round(map_long, 3)}')
        eval_results['mAP_very_long'] = float(f'{round(map_very_long, 3)}')
        return eval_results

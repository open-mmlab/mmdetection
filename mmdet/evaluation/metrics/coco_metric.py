# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmeval.detection import CocoMetric

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


def coco_metric_process(metric: CocoMetric, data_batch: dict,
                        data_samples: Sequence[dict]) -> None:
    """Process one batch of data samples and predictions. The processed results
    should be stored in ``self.results``, which will be used to compute the
    metrics when all batches have been processed.

    Args:
        data_batch (dict): A batch of data from the dataloader.
        data_samples (Sequence[dict]): A batch of data samples that
            contain annotations and predictions.
    """
    predictions = []
    groundtruths = []
    for data_sample in data_samples:
        result = dict()
        pred = data_sample['pred_instances']
        result['img_id'] = data_sample['img_id']
        result['bboxes'] = pred['bboxes'].cpu().numpy()
        result['scores'] = pred['scores'].cpu().numpy()
        result['labels'] = pred['labels'].cpu().numpy()
        # encode mask to RLE
        if 'masks' in pred:
            result['masks'] = encode_mask_results(
                pred['masks'].detach().cpu().numpy())
        # some detectors use different scores for bbox and mask
        if 'mask_scores' in pred:
            result['mask_scores'] = pred['mask_scores'].cpu().numpy()

        # parse gt
        gt = dict()
        gt['width'] = data_sample['ori_shape'][1]
        gt['height'] = data_sample['ori_shape'][0]
        gt['img_id'] = data_sample['img_id']
        if metric._coco_api is None:
            assert 'instances' in data_sample, \
                'ground truth is required for evaluation when ' \
                '`ann_file` is not provided'
            gt['instances'] = data_sample['instances']

        # pred, gt
        predictions.append(result)
        groundtruths.append(gt)
    metric.add(predictions, groundtruths)


CocoMetric.process = coco_metric_process

METRICS.register_module(module=CocoMetric)

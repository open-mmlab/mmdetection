# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import pycocotools.mask as maskUtils
from mmeval.detection import CocoMetric as MMEVAL_CocoMetric

from mmdet.registry import METRICS
from mmdet.structures.mask import BitmapMasks, encode_mask_results


@METRICS.register_module()
class CocoMetric(MMEVAL_CocoMetric):
    """
    Args:
        prefix (str): The prefix that will be added in the metric names to
            disambiguate homonymous metrics of different evaluators.
            Defaults to 'coco'.
    """

    def __init__(self, prefix: str = 'coco', **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix = prefix

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

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
            if self._coco_api is None:
                gt_instances = data_sample['gt_instances']

                labels = gt_instances['labels'].tolist()
                bboxes = gt_instances['bboxes'].tolist()
                gt_masks = gt_instances[
                    'masks'] if 'masks' in gt_instances else None

                instances = []
                for i in range(len(labels)):
                    instance = {'bbox_label': labels[i], 'bbox': bboxes[i]}

                    if gt_masks is not None:
                        if isinstance(gt_masks, BitmapMasks):
                            gt_mask = np.asfortranarray(gt_masks.masks[i])
                            gt_mask = maskUtils.encode(gt_mask)
                        else:
                            gt_mask = [
                                polygon.tolist()
                                for polygon in gt_masks.masks[i]
                            ]
                        instance['mask'] = gt_mask

                    instances.append(instance)
                gt['instances'] = instances

            # pred, gt
            predictions.append(result)
            groundtruths.append(gt)
        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        eval_results = self.compute(*args, **kwargs)

        eval_results = {
            f'{self.prefix}/{k}': v
            for k, v in eval_results.items()
        }

        return eval_results

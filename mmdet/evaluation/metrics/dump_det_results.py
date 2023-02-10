# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Sequence

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class DumpDetResults(DumpResults):
    """Dump model predictions to a pickle file for offline evaluation.

    Different from `DumpResults` in MMEngine, it compresses instance
    segmentation masks into RLE format.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            # remove gt
            data_sample.pop('gt_instances', None)
            data_sample.pop('ignored_instances', None)
            data_sample.pop('gt_panoptic_seg', None)

            if 'pred_instances' in data_sample:
                pred = data_sample['pred_instances']
                # encode mask to RLE
                if 'masks' in pred:
                    pred['masks'] = encode_mask_results(pred['masks'].numpy())
            if 'pred_panoptic_seg' in data_sample:
                warnings.warn(
                    'Panoptic segmentation map will not be compressed. '
                    'The dumped file will be extremely large! '
                    'Suggest using `CocoPanopticMetric` to save the coco '
                    'format json and segmentation png files directly.')
        self.results.extend(data_samples)

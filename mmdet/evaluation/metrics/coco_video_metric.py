# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Sequence

from mmengine.dist import broadcast_object_list, is_main_process

from mmdet.registry import METRICS
from .base_video_metric import collect_tracking_results
from .coco_metric import CocoMetric


@METRICS.register_module()
class CocoVideoMetric(CocoMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.
    """

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            ori_video_len = video_data_samples[0].ori_video_length
            video_len = len(video_data_samples)
            if ori_video_len == video_len:
                # video process
                for frame_id in range(video_len):
                    img_data_sample = video_data_samples[frame_id].to_dict()
                    super().process(None, [img_data_sample])
            else:
                # image process
                img_data_sample = video_data_samples[0].to_dict()
                super().process(None, [img_data_sample])

    def evaluate(self, size: int = 1) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        results = collect_tracking_results(self.results, self.collect_device)

        if is_main_process():
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]

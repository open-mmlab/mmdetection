# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence

from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.fileio import FileClient

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from .base_video_metric import collect_tracking_results
from .coco_metric import CocoMetric


@METRICS.register_module()
class CocoVideoMetric(CocoMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.
    """

    def __init__(self, ann_file: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file:
            file_client = FileClient.infer_client(uri=ann_file)
            with file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
        else:
            self._coco_api = None

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Note that we only modify ``pred['pred_instances']`` in ``CocoMetric``
        to ``pred['pred_det_instances']`` here.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            video_len = len(video_data_samples)

            for frame_id in range(video_len):
                img_data_sample = video_data_samples[frame_id].to_dict()
                result = dict()
                pred = img_data_sample['pred_instances']
                result['img_id'] = img_data_sample['img_id']
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
                gt['width'] = img_data_sample['ori_shape'][1]
                gt['height'] = img_data_sample['ori_shape'][0]
                gt['img_id'] = img_data_sample['img_id']
                if self._coco_api is None:
                    assert 'instances' in img_data_sample, \
                        'ground truth is required for evaluation when ' \
                        '`ann_file` is not provided'
                    gt['anns'] = img_data_sample['instances']
                # add converted result to the results list
                self.results.append((gt, result))

    def evaluate(self, size: int) -> dict:
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

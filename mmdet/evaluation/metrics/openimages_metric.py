# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np
from mmeval import OIDMeanAP
from mmengine.logging import MMLogger, print_log

from mmdet.registry import METRICS
from ..functional import eval_map


@METRICS.register_module()
class OpenImagesMetric(OIDMeanAP):
    """A wrapper of :class:`mmeval.OIDMeanAP`.

    This wrapper implements the `process` method that parses predictions and 
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty tabel of metrics per class.

    Args:
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.OIDMeanAP`.
    """

    def __init__(self, dist_backend: str = 'torch_cuda', **kwargs) -> None:
        ioa_thrs = kwargs.pop('ioa_thrs', None)
        if ioa_thrs is not None and 'iof_thrs' not in kwargs:
            kwargs['iof_thrs'] = ioa_thrs
            warnings.warn(
                'DeprecationWarning: The `ioa_thrs` parameter of '
                '`OpenImagesMetric` is deprecated, use `iof_thrs` instead!')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`OpenImagesMetric` is deprecated, use `dist_backend` instead.')

        super().__init__(
            classwise_results=True, dist_backend=dist_backend, **kwargs)

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Parse predictions and ground truths from ``data_samples`` and invoke
        ``self.add``.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            pred = {
                'bboxes': data_sample['pred_instances'].cpu().numpy(),
                'scores': data_sample['pred_instances'].cpu().numpy(),
                'labels': data_sample['pred_instances'].cpu().numpy()
            }
            predictions.append(pred)

            gt = {
                'instances': data_sample['instances'],
                'image_level_labels': data_sample.get('image_level_labels', None),  # noqa: E501
            }
            groundtruths.append(gt)

        self.add(predictions, groundtruths)
    
    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty tabel of metrics per class.
        
        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_results = metric_results['classwise_results']
        del metric_results['classwise_results']
        for i, (iou_thr, iof_thr) in enumerate(
                zip(self.iou_thrs, self.iof_thrs)):
            print_log(f'\n{"-" * 15}iou_thr, iof_thr:'
                      f' {iou_thr}, {iof_thr}{"-" * 15}')

        evaluate_results = {
            k: round(v * 100, 3) for k, v in metric_results.items()}
        return evaluate_results
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np
from mmeval import VOCMeanAP
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from ..functional import eval_map, eval_recalls


@METRICS.register_module()
class VOCMetric(VOCMeanAP):
    """A wrapper of :class:`mmeval.VOCMeanAP`.

    This wrapper implements the `process` method that parses predictions and 
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty tabel of metrics per class.

    Args:
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.VOCMeanAP`.
    """

    def __init__(self, dist_backend: str = 'torch_cuda', **kwargs) -> None:

        metric = kwargs.pop('metric', None)
        if metric is not None:
            warnings.warn(
                'DeprecationWarning: The `metric` parameter of '
                '`VOCMetric` is deprecated, only mAP is supported!')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`VOCMetric` is deprecated, use `dist_backend` instead.')

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
            gt = copy.deepcopy(data_sample)
            # TODO: Need to refactor to support LoadAnnotations
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            groundtruths.append(ann)

            pred = data_sample['pred_instances']
            pred['bboxes'] = pred['bboxes'].cpu().numpy()
            pred['scores'] = pred['scores'].cpu().numpy()
            pred['labels'] = pred['labels'].cpu().numpy()
            predictions.append(pred)

        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty tabel of metrics per class.
        
        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_results = metric_results['classwise_results']
        del metric_results['classwise_results']

        evaluate_results = {
            k: round(v * 100, 3) for k, v in metric_results.items()}
        return evaluate_results
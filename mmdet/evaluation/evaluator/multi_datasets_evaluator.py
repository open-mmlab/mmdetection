# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import warnings
from collections import OrderedDict
from typing import Any, Optional, Sequence, Union

import numpy as np
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import _to_cpu
from mmengine.registry import EVALUATOR
from mmengine.structures import BaseDataElement

from mmdet.utils import ConfigType


@EVALUATOR.register_module()
class MultiDatasetsEvaluator(Evaluator):
    """Wrapper class to compose class: `ConcatDataset` and multiple
    :class:`BaseMetric` instances.
    The metrics will be evaluated on each dataset slice separately. The name of
    the each metric is the concatenation of the dataset prefix, the metric
    prefix and the key of metric - e.g.
    `dataset_prefix/metric_prefix/accuracy`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
        dataset_prefixes (Sequence[str]): The prefix of each dataset. The
            length of this sequence should be the same as the length of the
            datasets.
    """

    def __init__(self, metrics: Union[ConfigType, BaseMetric, Sequence],
                 dataset_prefixes: Sequence[str]) -> None:
        super().__init__(metrics)
        self.dataset_prefixes = dataset_prefixes
        self._setups = False

    def _get_cumulative_sizes(self):
        # ConcatDataset have a property `cumulative_sizes`
        if isinstance(self.dataset_meta, Sequence):
            dataset_slices = self.dataset_meta[0]['cumulative_sizes']
            if not self._setups:
                self._setups = True
                for dataset_meta, metric in zip(self.dataset_meta,
                                                self.metrics):
                    metric.dataset_meta = dataset_meta
        else:
            dataset_slices = self.dataset_meta['cumulative_sizes']
        return dataset_slices

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        dataset_slices = self._get_cumulative_sizes()
        assert len(dataset_slices) == len(self.dataset_prefixes)

        for data, data_sample in zip(data_batch, data_samples):
            dataset_idx = data_sample.dataset_idx
            if isinstance(data_sample, BaseDataElement):
                self.metrics[dataset_idx].process([data],
                                                  [data_sample.to_dict()])
            else:
                self.metrics[dataset_idx].process([data], [data_sample])

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics_results = OrderedDict()
        dataset_slices = self._get_cumulative_sizes()
        assert len(dataset_slices) == len(self.dataset_prefixes)

        dataset_slices.insert(0, 0)
        dataset_slices = np.diff(dataset_slices).tolist()
        for dataset_prefix, dataset_slice, metric in zip(
                self.dataset_prefixes, dataset_slices, self.metrics):
            if len(metric.results) == 0:
                warnings.warn(
                    f'{metric.__class__.__name__} got empty `self.results`.'
                    'Please ensure that the processed results are properly '
                    'added into `self.results` in `process` method.')

            results = collect_results(metric.results, dataset_slice,
                                      metric.collect_device)

            if is_main_process():
                # cast all tensors in results list to cpu
                results = _to_cpu(results)
                _metrics = metric.compute_metrics(results)

                if metric.prefix:
                    final_prefix = '/'.join((dataset_prefix, metric.prefix))
                else:
                    final_prefix = dataset_prefix
                metric_results = {
                    '/'.join((final_prefix, k)): v
                    for k, v in _metrics.items()
                }

                # Check metric name conflicts
                for name in metric_results.keys():
                    if name in metrics_results:
                        raise ValueError(
                            'There are multiple evaluation results with '
                            f'the same metric name {name}. Please make '
                            'sure all metrics have different prefixes.')
                metrics_results.update(metric_results)
            metric.results.clear()
        if is_main_process():
            metrics_results = [metrics_results]
        else:
            metrics_results = [None]  # type: ignore
        broadcast_object_list(metrics_results)
        return metrics_results[0]

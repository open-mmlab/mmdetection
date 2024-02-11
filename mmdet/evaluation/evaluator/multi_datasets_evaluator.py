# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Sequence, Union

from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import _to_cpu
from mmengine.registry import EVALUATOR

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

        for dataset_prefix, start, end, metric in zip(
                self.dataset_prefixes, [0] + dataset_slices[:-1],
                dataset_slices, self.metrics):
            if len(metric.results) == 0:
                warnings.warn(
                    f'{metric.__class__.__name__} got empty `self.results`.'
                    'Please ensure that the processed results are properly '
                    'added into `self.results` in `process` method.')

            results = collect_results(metric.results, size,
                                      metric.collect_device)

            if is_main_process():
                # cast all tensors in results list to cpu
                results = _to_cpu(results)
                _metrics = metric.compute_metrics(
                    results[start:end])  # type: ignore

                if metric.prefix:
                    final_prefix = '/'.join((dataset_prefix, metric.prefix))
                else:
                    final_prefix = dataset_prefix
                print(f'================{final_prefix}================')
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

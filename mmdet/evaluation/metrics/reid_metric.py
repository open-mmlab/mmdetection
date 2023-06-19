# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class ReIDMetrics(BaseMetric):
    """mAP and CMC evaluation metrics for the ReID task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `mAP`.
        metric_options: (dict, optional): Options for calculating metrics.
            Allowed keys are 'rank_list' and 'max_rank'. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """
    allowed_metrics = ['mAP', 'CMC']
    default_prefix: Optional[str] = 'reid-metric'

    def __init__(self,
                 metric: Union[str, Sequence[str]] = 'mAP',
                 metric_options: Optional[dict] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')
        self.metrics = metrics

        self.metric_options = metric_options or dict(
            rank_list=[1, 5, 10, 20], max_rank=20)
        for rank in self.metric_options['rank_list']:
            assert 1 <= rank <= self.metric_options['max_rank']

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            pred_feature = data_sample['pred_feature']
            assert isinstance(pred_feature, torch.Tensor)
            gt_label = data_sample.get('gt_label', data_sample['gt_label'])
            assert isinstance(gt_label['label'], torch.Tensor)
            result = dict(
                pred_feature=pred_feature.data.cpu(),
                gt_label=gt_label['label'].cpu())
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        pids = torch.cat([result['gt_label'] for result in results]).numpy()
        features = torch.stack([result['pred_feature'] for result in results])

        n, c = features.size()
        mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        distmat = mat + mat.t()
        distmat.addmm_(features, features.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()

        indices = np.argsort(distmat, axis=1)
        matches = (pids[indices] == pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        num_valid_q = 0.
        for q_idx in range(n):
            # remove self
            raw_cmc = matches[q_idx][1:]
            if not np.any(raw_cmc):
                # this condition is true when query identity
                # does not appear in gallery
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:self.metric_options['max_rank']])
            num_valid_q += 1.

            # compute average precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, \
            'Error: all query identities do not appear in gallery'

        all_cmc = np.asarray(all_cmc)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        if 'mAP' in self.metrics:
            metrics['mAP'] = np.around(mAP, decimals=3)
        if 'CMC' in self.metrics:
            for rank in self.metric_options['rank_list']:
                metrics[f'R{rank}'] = np.around(all_cmc[rank - 1], decimals=3)

        return metrics

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from mmcv import imwrite
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None

from mmdet.registry import METRICS


@METRICS.register_module()
class SemSegMetric(BaseMetric):
    """mIoU evaluation metric.

    Args:
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_metrics: Sequence[str] = ['mIoU'],
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 backend_args: dict = None,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(iou_metrics, str):
            iou_metrics = [iou_metrics]
        if not set(iou_metrics).issubset(set(['mIoU', 'mDice', 'mFscore'])):
            raise KeyError(f'metrics {iou_metrics} is not supported. '
                           f'Only supports mIoU/mDice/mFscore.')
        self.metrics = iou_metrics
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.backend_args = backend_args

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['sem_seg'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['sem_seg'].squeeze().to(
                    pred_label)
                ignore_index = data_sample['pred_sem_seg'].get(
                    'ignore_index', 255)
                self.results.append(
                    self._compute_pred_stats(pred_label, label, num_classes,
                                             ignore_index))

            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                output = Image.fromarray(output_mask.astype(np.uint8))
                imwrite(output, png_filename, backend_args=self.backend_args)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        ret_metrics = self.get_return_metrics(results)

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        print_semantic_table(ret_metrics, self.dataset_meta['classes'], logger)

        return metrics

    def _compute_pred_stats(self, pred_label: torch.tensor,
                            label: torch.tensor, num_classes: int,
                            ignore_index: int):
        """Parse semantic segmentation predictions.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        assert pred_label.shape == label.shape
        mask = label != ignore_index
        label, pred_label = label[mask], pred_label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_label = torch.histc(
            label.float(), bins=num_classes, min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        result = dict(
            area_intersect=area_intersect,
            area_union=area_union,
            area_pred_label=area_pred_label,
            area_label=area_label)
        return result

    def get_return_metrics(self, results: list) -> dict:
        """Calculate evaluation metrics.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        total_area_intersect = sum([r['area_intersect'] for r in results])
        total_area_union = sum([r['area_union'] for r in results])
        total_area_pred_label = sum([r['area_pred_label'] for r in results])
        total_area_label = sum([r['area_label'] for r in results])

        all_acc = total_area_intersect / total_area_label
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in self.metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], self.beta)
                    for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.cpu().numpy()
            for metric, value in ret_metrics.items()
        }

        return ret_metrics


def print_semantic_table(
        results: dict,
        class_names: list,
        logger: Optional[Union['MMLogger', str]] = None) -> None:
    """Print semantic segmentation evaluation results table.

    Args:
        results (dict): The evaluation results.
        class_names (list): Class names.
        logger (MMLogger | str, optional): Logger used for printing.
            Default: None.
    """
    # each class table
    results.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in results.items()
    })

    print_log('per class results:', logger)
    if PrettyTable:
        class_table_data = PrettyTable()
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        print_log('\n' + class_table_data.get_string(), logger=logger)
    else:
        logger.warning(
            '`prettytable` is not installed, for better table format, '
            'please consider installing it with "pip install prettytable"')
        print_result = {}
        for class_name, iou, acc in zip(class_names, ret_metrics_class['IoU'],
                                        ret_metrics_class['Acc']):
            print_result[class_name] = {'IoU': iou, 'Acc': acc}
        print_log(print_result, logger)

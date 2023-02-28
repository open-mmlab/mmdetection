# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
import warnings
from typing import List, Optional, Sequence, Union

from mmengine.logging import MMLogger, print_log
from mmeval import LVISDetection
from terminaltables import AsciiTable
from torch import Tensor

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class LVISMetric(LVISDetection):
    """LVIS evaluation metric.

    Args:
        ann_file (str, optional): Path to the COCO LVIS format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to COCO LVIS format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (int): Numbers of proposals to be evaluated.
            Defaults to 300.
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    default_prefix: Optional[str] = 'lvis'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: int = 300,
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`LVISMetric` is deprecated, use `dist_backend` instead.')
        logger = MMLogger.get_current_instance()

        super().__init__(
            ann_file=ann_file,
            metric=metric,
            iou_thrs=iou_thrs,
            classwise=classwise,
            proposal_nums=proposal_nums,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            backend_args=backend_args,
            dist_backend=dist_backend,
            logger=logger,
            **kwargs)
        self.prefix = prefix or self.default_prefix

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            pred = dict()
            pred_instances = data_sample['pred_instances']
            pred['img_id'] = data_sample['img_id']
            pred['bboxes'] = pred_instances['bboxes'].cpu().numpy()
            pred['scores'] = pred_instances['scores'].cpu().numpy()
            pred['labels'] = pred_instances['labels'].cpu().numpy()
            if 'masks' in pred_instances:
                pred['masks'] = encode_mask_results(
                    pred_instances['masks'].detach().cpu().numpy(
                    )) if isinstance(pred_instances['masks'],
                                     Tensor) else pred_instances['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred_instances:
                pred['mask_scores'] = \
                    pred_instances['mask_scores'].cpu().numpy()
            predictions.append(pred)
            # LVIS only supports loading annotation from JSON
            ann = dict()  # create dummy ann
            groundtruths.append(ann)
        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        if self.format_only:
            print_log(
                'Results are saved in '
                f'{osp.dirname(self.outfile_prefix)}',
                logger='current')
            return metric_results
        for metric in self.metrics:
            result = metric_results.pop(f'{metric}_result')
            if metric == 'proposal':
                table_title = '  Recall Results (%)'
                if self.metric_items is None:
                    assert len(result) == 4
                    headers = [
                        f'AR@{self.proposal_nums}',
                        f'AR_s@{self.proposal_nums}',
                        f'AR_m@{self.proposal_nums}',
                        f'AR_l@{self.proposal_nums}'
                    ]
                else:
                    assert len(result) == len(self.metric_items)
                    headers = self.metric_items
            else:
                table_title = f' {metric} Results (%)'
                if self.metric_items is None:
                    assert len(result) == 9
                    headers = [
                        f'{metric}_AP', f'{metric}_AP50', f'{metric}_AP75',
                        f'{metric}_APs', f'{metric}_APm', f'{metric}_APl',
                        f'{metric}_APr', f'{metric}_APc', f'{metric}_APf'
                    ]
                else:
                    assert len(result) == len(self.metric_items)
                    headers = [
                        f'{metric}_{item}' for item in self.metric_items
                    ]
            table_data = [headers, result]
            table = AsciiTable(table_data, title=table_title)
            print_log('\n' + table.table, logger='current')

            if self.classwise and \
                    f'{metric}_classwise_result' in metric_results:
                print_log(
                    f'Evaluating {metric} metric of each category...',
                    logger='current')
                classwise_table_title = f' {metric} Classwise Results (%)'
                classwise_result = metric_results.pop(
                    f'{metric}_classwise_result')

                num_columns = min(6, len(classwise_result) * 2)
                results_flatten = list(itertools.chain(*classwise_result))
                headers = ['category', f'{metric}_AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data, title=classwise_table_title)
                print_log('\n' + table.table, logger='current')
        evaluate_results = {
            f'{self.prefix}/{k}(%)': round(float(v) * 100, 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

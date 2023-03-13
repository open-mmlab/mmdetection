# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence, Union

from mmengine.logging import MMLogger
from mmeval import LVISDetection
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
        predictions = []
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
        self.add(predictions)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        if self.format_only:
            return metric_results

        evaluate_results = {
            f'{self.prefix}/{k}(%)': round(float(v) * 100, 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

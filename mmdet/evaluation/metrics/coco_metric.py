# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger
from mmeval import COCODetection
from torch import Tensor

from mmdet.registry import METRICS
from mmdet.structures.mask import (BitmapMasks, PolygonMasks,
                                   encode_mask_results)


@METRICS.register_module()
class CocoMetric(COCODetection):
    """A wrapper of :class:`mmeval.COCODetection`.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', and 'proposal'. Defaults to 'bbox'.
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        classwise (bool):Whether to return the computed
            results of each class. Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (1, 10, 100).
            Note: it defaults to (100, 300, 1000) in MMDet 2.x.
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
        gt_mask_area (bool): Whether calculate GT mask area when not loading
            ann_file. If True, the GT instance area will be the mask area,
            else the bounding box area. It will not be used when loading
            ann_file. Defaults to True.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (1, 10, 100),
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 gt_mask_area: bool = True,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`CocoMetric` is deprecated, use `dist_backend` instead.')

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
            gt_mask_area=gt_mask_area,
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

            # parse gt
            if self._coco_api is None:
                ann = self.add_gt(data_sample)
            else:
                ann = dict()
            groundtruths.append(ann)

        self.add(predictions, groundtruths)

    @staticmethod
    def add_gt(data_sample):
        ann = dict()
        ann['width'] = data_sample['ori_shape'][1]
        ann['height'] = data_sample['ori_shape'][0]
        ann['img_id'] = data_sample['img_id']

        gt_instances = data_sample['gt_instances']
        ignored_instances = data_sample['ignored_instances']

        ann['bboxes'] = np.concatenate(
            (gt_instances['bboxes'].cpu().numpy(),
             ignored_instances['bboxes'].cpu().numpy()),
            axis=0)
        ann['labels'] = np.concatenate(
            (gt_instances['labels'].cpu().numpy(),
             ignored_instances['labels'].cpu().numpy()),
            axis=0)
        ann['ignore_flags'] = np.concatenate(
            (np.zeros(len(gt_instances['labels'])),
             np.ones(len(ignored_instances['labels']))),
            axis=0)
        assert len(ann['bboxes']) == len(ann['labels'])
        if 'masks' in gt_instances:
            assert isinstance(gt_instances['masks'],
                              (PolygonMasks, BitmapMasks)) and \
                   isinstance(ignored_instances['masks'],
                              (PolygonMasks, BitmapMasks))
            ann['masks']: list = []
            ann['masks'].extend(
                encode_mask_results(gt_instances['masks'].to_ndarray()))
            ann['masks'].extend(
                encode_mask_results(ignored_instances['masks'].to_ndarray()))
            assert len(ann['bboxes']) == len(ann['masks'])
        return ann

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

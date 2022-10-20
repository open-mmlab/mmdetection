# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import List, Optional, Sequence, Union

import numpy as np
from mmeval import CocoDetectionMetric as _CocoMetric
from terminaltables import AsciiTable

from mmdet.registry import METRICS
from mmdet.structures.mask import (BitmapMasks, PolygonMasks,
                                   encode_mask_results)


@METRICS.register_module()
class CocoMetric(_CocoMetric):
    """COCO object detection task evaluation metric.

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
        classwise_result (bool):Whether to return the computed
            results of each class. Defaults to False.
        proposal_nums (Sequence[int]nig): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmeval.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        gt_mask_area (bool): Whether calculate GT mask area when not loading
            ann_file. If True, the GT instance area will be the mask area,
            else the bounding box area. It will not be used when loading
            ann_file. Defaults to True.
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
                 classwise_result: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 gt_mask_area: bool = True,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            iou_thrs=iou_thrs,
            classwise_result=classwise_result,
            proposal_nums=proposal_nums,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            file_client_args=file_client_args,
            gt_mask_area=gt_mask_area,
            dist_backend=dist_backend,
            **kwargs)

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
                    pred_instances['masks'].detach().cpu().numpy())
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
        from mmengine.logging import print_log
        for metric in self.metrics:
            print_log(f'Evaluating {metric}...', logger='current')

            try:
                result = metric_results.pop(f'{metric}_result')
            except KeyError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger='current')
                break

            assert len(result) == 6
            if metric == 'proposal':
                headers = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
            else:
                headers = [
                    f'{metric}_mAP', f'{metric}_mAP_50', f'{metric}_mAP_75',
                    f'{metric}_mAP_s', f'{metric}_mAP_m', f'{metric}_mAP_l'
                ]
            table_data = [headers, result]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger='current')

            if self.classwise_result and \
                    f'{metric}_classwise_result' in metric_results:
                print_log(
                    f'Evaluating {metric} metric of each category...',
                    logger='current')

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
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger='current')

        return metric_results

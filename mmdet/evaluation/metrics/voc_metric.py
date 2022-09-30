# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.logging import print_log
from mmeval import VOCMeanAP
from terminaltables import AsciiTable

from mmdet.registry import METRICS


@METRICS.register_module()
class VOCMetric(VOCMeanAP):
    """A wrapper of :class:`mmeval.VOCMeanAP`.

    This wrapper implements the `process` method that parses predictions and
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty table of metrics per class.

    Args:
        iou_thrs (float ｜ List[float]): IoU thresholds. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        num_classes (int, optional): The number of classes. If None, it will be
            obtained from the 'CLASSES' field in ``self.dataset_meta``.
            Defaults to None.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'.
            Defaults to 'area'.
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to True.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        drop_class_ap (bool): Whether to drop the class without ground truth
            when calculating the average precision for each class.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`.
    """

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[Tuple]] = None,
                 num_classes: Optional[int] = None,
                 eval_mode: str = 'area',
                 use_legacy_coordinate: bool = True,
                 nproc: int = 4,
                 drop_class_ap: bool = True,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:

        metric = kwargs.pop('metric', None)
        if metric is not None:
            warnings.warn('DeprecationWarning: The `metric` parameter of '
                          '`VOCMetric` is deprecated, only mAP is supported!')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`VOCMetric` is deprecated, use `dist_backend` instead.')

        super().__init__(
            iou_thrs=iou_thrs,
            scale_ranges=scale_ranges,
            num_classes=num_classes,
            eval_mode=eval_mode,
            use_legacy_coordinate=use_legacy_coordinate,
            nproc=nproc,
            classwise_result=True,
            drop_class_ap=drop_class_ap,
            dist_backend=dist_backend,
            **kwargs)

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
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        classwise_result = metric_results['classwise_result']
        del metric_results['classwise_result']

        classes = self.dataset_meta['CLASSES']
        header = ['class', 'gts', 'dets', 'recall', 'ap']

        for i, iou_thr in enumerate(self.iou_thrs):
            for j, scale_range in enumerate(self.scale_ranges):
                table_title = f' IoU thr: {iou_thr} '
                if scale_range != (None, None):
                    table_title += f'Scale range: {scale_range} '

                table_data = [header]
                aps = []
                for k in range(len(classes)):
                    class_results = classwise_result[k]
                    recalls = class_results['recalls'][i, j]
                    recall = 0 if len(recalls) == 0 else recalls[-1]
                    row_data = [
                        classes[k], class_results['num_gts'][i, j],
                        class_results['num_dets'],
                        round(recall, 3),
                        round(class_results['ap'][i, j], 3)
                    ]
                    table_data.append(row_data)
                    if class_results['num_gts'][i, j] > 0:
                        aps.append(class_results['ap'][i, j])

                mean_ap = np.mean(aps) if aps != [] else 0
                table_data.append(['mAP', '', '', '', f'{mean_ap:.3f}'])
                table = AsciiTable(table_data, title=table_title)
                table.inner_footing_row_border = True
                print_log('\n' + table.table, logger='current')

        evaluate_results = {
            f'pascal_voc/{k}': round(float(v), 3)
            for k, v in metric_results.items()
        }
        return evaluate_results

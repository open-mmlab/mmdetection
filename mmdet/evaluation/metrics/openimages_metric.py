# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.logging import print_log
from mmeval import OIDMeanAP
from terminaltables import AsciiTable

from mmdet.registry import METRICS


@METRICS.register_module()
class OpenImagesMetric(OIDMeanAP):
    """A wrapper of :class:`mmeval.OIDMeanAP`.

    This wrapper implements the `process` method that parses predictions and
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty table of metrics per class.

    Args:
        iof_thrs (float ｜ List[float]): IoF thresholds. Defaults to 0.5.
        iou_thrs (float ｜ List[float]): IoU thresholds. Defaults to 0.5.
        use_group_of (bool): Whether consider group of groud truth
            bboxes during evaluating. Defaults to True.
        get_supercategory (bool, optional): Whether to get parent class of the
            current class. Defaults to True.
        filter_labels (bool, optional): Whether filter unannotated classes.
            Defaults to True.
        class_relation_matrix (numpy.ndarray, optional): The matrix of the
            corresponding relationship between the parent class and the child
            class. If None, it will be obtained from the 'RELATION_MATRIX'
            field in ``self.dataset_meta``. Defaults to None.
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
            Defaults to False.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        drop_class_ap (bool): Whether to drop the class without ground truth
            when calculating the average precision for each class.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`.
    """
    default_prefix: Optional[str] = 'openimages'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 iof_thrs: Union[float, List[float]] = 0.5,
                 use_group_of: bool = True,
                 get_supercategory: bool = True,
                 filter_labels: bool = True,
                 class_relation_matrix: Optional[np.ndarray] = None,
                 scale_ranges: Optional[List[Tuple]] = None,
                 num_classes: Optional[int] = None,
                 eval_mode: str = 'area',
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 drop_class_ap: bool = True,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`ProposalRecallMetric` is deprecated, '
                'use `dist_backend` instead.')

        ioa_thrs = kwargs.pop('ioa_thrs', None)
        if ioa_thrs is not None:
            iof_thrs = ioa_thrs
            warnings.warn(
                'DeprecationWarning: The `ioa_thrs` parameter of '
                '`OpenImagesMetric` is deprecated, use `iof_thrs` instead!')

        super().__init__(
            iou_thrs=iou_thrs,
            iof_thrs=iof_thrs,
            use_group_of=use_group_of,
            get_supercategory=get_supercategory,
            filter_labels=filter_labels,
            class_relation_matrix=class_relation_matrix,
            scale_ranges=scale_ranges,
            num_classes=num_classes,
            eval_mode=eval_mode,
            use_legacy_coordinate=use_legacy_coordinate,
            nproc=nproc,
            classwise=True,  # should force set `classwise=True`
            drop_class_ap=drop_class_ap,
            dist_backend=dist_backend,
            **kwargs)

        self.prefix = prefix or self.default_prefix

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
            pred = {
                'bboxes': data_sample['pred_instances']
                ['bboxes'].cpu().numpy(),  # noqa: E501
                'scores': data_sample['pred_instances']
                ['scores'].cpu().numpy(),  # noqa: E501
                'labels':
                data_sample['pred_instances']['labels'].cpu().numpy()
            }
            predictions.append(pred)

            gt = {
                'instances': data_sample['instances'],
                'image_level_labels': data_sample.get('image_level_labels',
                                                      None),  # noqa: E501
            }
            groundtruths.append(gt)

        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        assert 'classwise_result' in metric_results
        classwise_result = metric_results.pop('classwise_result')

        classes = self.dataset_meta['classes']
        header = ['class', 'gts', 'dets', 'recall(%)', 'ap(%)']

        for i, (iou_thr,
                iof_thr) in enumerate(zip(self.iou_thrs,
                                          self.iof_thrs)):  # noqa: E501
            for j, scale_range in enumerate(self.scale_ranges):
                table_title = ' Bbox Results ' \
                              f'(IoU thr={iou_thr}, IoF thr={iof_thr}) '
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
                        round(recall * 100, 2),
                        round(class_results['ap'][i, j] * 100, 2)
                    ]
                    table_data.append(row_data)
                    if class_results['num_gts'][i, j] > 0:
                        aps.append(class_results['ap'][i, j])

                mean_ap = np.mean(aps) if aps != [] else 0
                table_data.append(
                    ['mAP', '', '', '', f'{round(mean_ap * 100, 2)}'])
                table = AsciiTable(table_data, title=table_title)
                table.inner_footing_row_border = True
                print_log('\n' + table.table, logger='current')

        evaluate_results = {
            f'{self.prefix}/{k}(%)': round(float(v) * 100, 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

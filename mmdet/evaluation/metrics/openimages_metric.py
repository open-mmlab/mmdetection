# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet.registry import METRICS
from ..functional import eval_map


@METRICS.register_module()
class OpenImagesMetric(BaseMetric):
    """OpenImages evaluation metric.

    Evaluate detection mAP for OpenImages. Please refer to
    https://storage.googleapis.com/openimages/web/evaluation.html for more
    details.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        ioa_thrs (float or List[float]): IoA threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None
        use_group_of (bool): Whether consider group of groud truth bboxes
            during evaluating. Defaults to True.
        get_supercategory (bool): Whether to get parent class of the
            current class. Default: True.
        filter_labels (bool): Whether filter unannotated classes.
            Default: True.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'openimages'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 ioa_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 use_group_of: bool = True,
                 get_supercategory: bool = True,
                 filter_labels: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) else iou_thrs
        self.ioa_thrs = [ioa_thrs] if (isinstance(ioa_thrs, float)
                                       or ioa_thrs is None) else ioa_thrs
        assert isinstance(self.iou_thrs, list) and isinstance(
            self.ioa_thrs, list)
        assert len(self.iou_thrs) == len(self.ioa_thrs)

        self.scale_ranges = scale_ranges
        self.use_group_of = use_group_of
        self.get_supercategory = get_supercategory
        self.filter_labels = filter_labels

    def _get_supercategory_ann(self, instances: List[dict]) -> List[dict]:
        """Get parent classes's annotation of the corresponding class.

        Args:
            instances (List[dict]): A list of annotations of the instances.

        Returns:
            List[dict]: Annotations extended with super-category.
        """
        supercat_instances = []
        relation_matrix = self.dataset_meta['RELATION_MATRIX']
        for instance in instances:
            labels = np.where(relation_matrix[instance['bbox_label']])[0]
            for label in labels:
                if label == instance['bbox_label']:
                    continue
                new_instance = copy.deepcopy(instance)
                new_instance['bbox_label'] = label
                supercat_instances.append(new_instance)
        return supercat_instances

    def _process_predictions(self, pred_bboxes: np.ndarray,
                             pred_scores: np.ndarray, pred_labels: np.ndarray,
                             gt_instances: list,
                             image_level_labels: np.ndarray) -> tuple:
        """Process results of the corresponding class of the detection bboxes.

        Note: It will choose to do the following two processing according to
        the parameters:

        1. Whether to add parent classes of the corresponding class of the
        detection bboxes.

        2. Whether to ignore the classes that unannotated on that image.

        Args:
            pred_bboxes (np.ndarray): bboxes predicted by the model
            pred_scores (np.ndarray): scores predicted by the model
            pred_labels (np.ndarray): labels predicted by the model
            gt_instances (list): ground truth annotations
            image_level_labels (np.ndarray): human-verified image level labels

        Returns:
            tuple: Processed bboxes, scores, and labels.
        """
        processed_bboxes = copy.deepcopy(pred_bboxes)
        processed_scores = copy.deepcopy(pred_scores)
        processed_labels = copy.deepcopy(pred_labels)
        gt_labels = np.array([ins['bbox_label'] for ins in gt_instances],
                             dtype=np.int64)
        if image_level_labels is not None:
            allowed_classes = np.unique(
                np.append(gt_labels, image_level_labels))
        else:
            allowed_classes = np.unique(gt_labels)
        relation_matrix = self.dataset_meta['RELATION_MATRIX']
        pred_classes = np.unique(pred_labels)
        for pred_class in pred_classes:
            classes = np.where(relation_matrix[pred_class])[0]
            for cls in classes:
                if (cls in allowed_classes and cls != pred_class
                        and self.get_supercategory):
                    # add super-supercategory preds
                    index = np.where(pred_labels == pred_class)[0]
                    processed_scores = np.concatenate(
                        [processed_scores, pred_scores[index]])
                    processed_bboxes = np.concatenate(
                        [processed_bboxes, pred_bboxes[index]])
                    extend_labels = np.full(index.shape, cls, dtype=np.int64)
                    processed_labels = np.concatenate(
                        [processed_labels, extend_labels])
                elif cls not in allowed_classes and self.filter_labels:
                    # remove unannotated preds
                    index = np.where(processed_labels != cls)[0]
                    processed_scores = processed_scores[index]
                    processed_bboxes = processed_bboxes[index]
                    processed_labels = processed_labels[index]
        return processed_bboxes, processed_scores, processed_labels

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
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            # add super-category instances
            # TODO: Need to refactor to support LoadAnnotations
            instances = gt['instances']
            if self.get_supercategory:
                supercat_instances = self._get_supercategory_ann(instances)
                instances.extend(supercat_instances)
            gt_labels = []
            gt_bboxes = []
            is_group_ofs = []
            for ins in instances:
                gt_labels.append(ins['bbox_label'])
                gt_bboxes.append(ins['bbox'])
                is_group_ofs.append(ins['is_group_of'])
            ann = dict(
                labels=np.array(gt_labels, dtype=np.int64),
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4)),
                gt_is_group_ofs=np.array(is_group_ofs, dtype=bool))

            image_level_labels = gt.get('image_level_labels', None)
            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            pred_bboxes, pred_scores, pred_labels = self._process_predictions(
                pred_bboxes, pred_scores, pred_labels, instances,
                image_level_labels)

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)
            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()
        # get dataset type
        dataset_type = self.dataset_meta.get('dataset_type')
        if dataset_type not in ['oid_challenge', 'oid_v6']:
            dataset_type = 'oid_v6'
            print_log(
                'Cannot infer dataset type from the length of the'
                ' classes. Set `oid_v6` as dataset type.',
                logger='current')
        mean_aps = []
        for i, (iou_thr,
                ioa_thr) in enumerate(zip(self.iou_thrs, self.ioa_thrs)):
            if self.use_group_of:
                assert ioa_thr is not None, 'ioa_thr must have value when' \
                                            ' using group_of in evaluation.'
            print_log(f'\n{"-" * 15}iou_thr, ioa_thr: {iou_thr}, {ioa_thr}'
                      f'{"-" * 15}')
            mean_ap, _ = eval_map(
                preds,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                ioa_thr=ioa_thr,
                dataset=dataset_type,
                logger=logger,
                use_group_of=self.use_group_of)

            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results

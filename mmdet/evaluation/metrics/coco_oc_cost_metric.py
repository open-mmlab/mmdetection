# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path
from mmengine.logging import MMLogger
from scipy.spatial.distance import cdist

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.mask import encode_mask_results

try:
    import ot
except ImportError:
    ot = None


@METRICS.register_module()
class CocoOCCostMetric(BaseMetric):
    """Coco OC-Cost evaluation metric.

    First, please install POT `pip install POT`.
    Evaluate OC-Cost for detection tasks including instance segmentation.
    Please refer to https://arxiv.org/abs/2203.14438 for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        alpha (float): Balancing localization and classification costs.
            lambda in the paper. Defaults to 0.5.
        beta (float): Cost of extra / missing detections. Defaults to 0.6.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm'. Defaults to 'bbox'.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
    """

    default_prefix: str | None = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 alpha: float = 0.5,
                 beta: float = 0.6,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if ot is None:
            raise RuntimeError('POT is not installed')

        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.alpha = alpha
        self.beta = beta
        allowed_metrics = ['bbox', 'segm']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                msg = "metric should be one of 'bbox', 'segm' but "
                f'got {metric}.'
                raise KeyError(msg, )

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Copied from mmdetection/blob/main/mmdet/datasets/coco.py.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".

        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process data batch.

        Copied from mmdetection/blob/main/mmdet/datasets/coco.py.
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def _cls_cost_func(self, gt_label_score: np.ndarray,
                       pred_label_score: np.ndarray) -> float:
        """Calculate classification cost.

        Args:
            gt_label_score (np.ndarray): Ground truth label and score. [l, s].
            pred_label_score (np.ndarray): Prediction label and score. [l, s].

        Returns:
            float: a unit cost
        """
        if gt_label_score[0] == pred_label_score[0]:
            cls_cost = np.abs(gt_label_score[1] - pred_label_score[1])
        else:
            cls_cost = gt_label_score[1] + pred_label_score[1]
        cls_cost *= 0.5  # normalized to [0, 1]

        return cls_cost

    def eval_oc_cost(self, predictions: list[dict], metric: str) -> float:
        """Evaluate optimal transportation cost.

        Args:
            predictions (list[dict]): The processed results of each batch.
            metric (str): Metric name, 'bbox' or 'segm'.

        Returns:
            float: Mean optimal transportation cost cost.
        """
        ot_costs = []
        for img_id, prediction in zip(self.img_ids, predictions):
            # prepare GT
            img_info = self._coco_api.load_imgs(img_id)[0]
            ann_ids = self._coco_api.get_ann_ids(img_ids=img_id)
            ann_info = self._coco_api.load_anns(ann_ids)
            gt = []
            gt_label_score = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                if metric == 'bbox':
                    x1, y1, w, h = ann['bbox']
                    gt.append([x1, y1, x1 + w, y1 + h])
                elif metric == 'segm':
                    if isinstance(ann['segmentation'], Sequence):
                        # convert Sequence to RLE
                        rle = mask_utils.frPyObjects(ann['segmentation'],
                                                     img_info['height'],
                                                     img_info['width'])[0]
                    else:
                        rle = ann['segmentation']
                    gt.append(rle)
                gt_label_score.append([ann['category_id'], 1])

            # prepare prediction
            pred_label_score = np.concatenate([
                prediction['labels'][..., None], prediction['scores'][...,
                                                                      None]
            ],
                                              axis=1)
            pred = (
                prediction['bboxes']
                if metric == 'bbox' else prediction['masks'])

            n = len(gt)
            m = len(pred)

            if n == 0 and m == 0:
                ot_costs.append(0)
                continue

            if n == 0 or m == 0:
                ot_costs.append(self.beta)
                continue

            # calculate OT cost
            if metric == 'bbox':
                iou_val = bbox_overlaps(
                    torch.Tensor(gt)[None, :, :4],
                    torch.Tensor(pred)[None, :, :4],
                    mode='giou').numpy()
                loc_cost = 1 - (iou_val + 1) * 0.5  # normalized to [0, 1]
            elif metric == 'segm':
                loc_cost = 1 - mask_utils.iou(gt, pred, [0] * len(pred))

            cls_cost = cdist(gt_label_score, pred_label_score,
                             self._cls_cost_func)

            cost_matrix = np.zeros((n + 1, m + 1))
            cost_matrix[:n, :m] = self.alpha * loc_cost + (
                1 - self.alpha) * cls_cost

            dist_a = np.ones(n + 1)
            dist_b = np.ones(m + 1)

            # cost for dummy demander / supplier
            cost_matrix[-1, :] = self.beta
            cost_matrix[:, -1] = self.beta
            dist_a[-1] = m
            dist_b[-1] = n

            ot_matrix = ot.emd(dist_a, dist_b, cost_matrix)

            # post process
            ot_matrix[-1, -1] = 0
            ot_matrix /= ot_matrix.sum()
            ot_cost = (cost_matrix * ot_matrix).sum()

            ot_costs.append(ot_cost)

        return np.mean(ot_costs)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        eval_results = OrderedDict()

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            mean_ot_costs = self.eval_oc_cost(preds, metric)

            eval_results[f'{metric}_OCCost'] = float(
                f'{round(mean_ot_costs, 5)}')
            logger.info(f'{metric}_OCCost: {mean_ot_costs:.5f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

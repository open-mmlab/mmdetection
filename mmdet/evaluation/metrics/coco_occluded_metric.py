# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence, Union

from mmengine.logging import MMLogger
from mmeval import CocoOccludedSeparated

from mmdet.registry import METRICS
from .coco_metric import parse_coco_groundtruth, parse_coco_prediction


@METRICS.register_module()
class CocoOccludedSeparatedMetric(CocoOccludedSeparated):
    """A wrapper of :class:`mmeval.CocoOccludedSeparated`.

    Metric of separated and occluded masks which presented in paper `A Tri-
    Layer Plugin to Improve Occluded Detection.
    <https://arxiv.org/abs/2210.10046>`_.

    Separated COCO and Occluded COCO are automatically generated subsets of
    COCO val dataset, collecting separated objects and partially occluded
    objects for a large variety of categories. In this way, we define
    occlusion into two major categories: separated and partially occluded.

    - Separation: target object segmentation mask is separated into distinct
      regions by the occluder.
    - Partial Occlusion: target object is partially occluded but the
      segmentation mask is connected.

    These two new scalable real-image datasets are to benchmark a model's
    capability to detect occluded objects of 80 common categories.

    Please cite the paper if you use this dataset:

    @article{zhan2022triocc,
        title={A Tri-Layer Plugin to Improve Occluded Detection},
        author={Zhan, Guanqi and Xie, Weidi and Zisserman, Andrew},
        journal={British Machine Vision Conference},
        year={2022}
    }

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', and 'proposal'.
            Defaults to ['bbox', 'segm'].
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
        occluded_ann (str): Path to the occluded coco annotation file.
        separated_ann (str): Path to the separated coco annotation file.
        score_thr (float): Score threshold of the detection masks.
            Defaults to 0.3.
        iou_thr (float): IoU threshold for the recall calculation.
            Defaults to 0.75.
        **kwargs: Keyword parameters passed to :class:`CocoOccludedSeparated`.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(
            self,
            ann_file: Optional[str] = None,
            metric: Union[str, List[str]] = ['bbox', 'segm'],
            iou_thrs: Optional[Union[float, Sequence[float]]] = None,
            classwise: bool = False,
            proposal_nums: Sequence[int] = (1, 10, 100),
            metric_items: Optional[Sequence[str]] = None,
            format_only: bool = False,
            outfile_prefix: Optional[str] = None,
            backend_args: Optional[dict] = None,
            gt_mask_area: bool = True,
            occluded_ann:
        str = 'https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/occluded_coco.pkl',  # noqa
            separated_ann:
        str = 'https://www.robots.ox.ac.uk/~vgg/research/tpod/datasets/separated_coco.pkl',  # noqa
            score_thr: float = 0.3,
            iou_thr: float = 0.75,
            prefix: Optional[str] = None,
            dist_backend: str = 'torch_cuda',
            **kwargs) -> None:

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`CocoOccludedSeparatedMetric` is deprecated, '
                'use `dist_backend` instead.')

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
            occluded_ann=occluded_ann,
            separated_ann=separated_ann,
            score_thr=score_thr,
            iou_thr=iou_thr,
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
            # parse prediction
            pred = parse_coco_prediction(data_sample)
            predictions.append(pred)

            # parse groundtruth
            if self._coco_api is None:
                ann = parse_coco_groundtruth(data_sample)
            else:
                ann = dict()
            groundtruths.append(ann)

        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()

        evaluate_results = {
            f'{self.prefix}/{k}(%)': round(float(v) * 100, 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

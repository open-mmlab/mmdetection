# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

from mmengine.logging import MMLogger
from mmeval import CocoPanoptic

from mmdet.registry import METRICS

try:
    import panopticapi
except ImportError:
    panopticapi = None


@METRICS.register_module()
class CocoPanopticMetric(CocoPanoptic):
    """A wrapper of :class:`mmeval.CocoPanoptic`.

    This wrapper implements the `process` method that parses predictions and
    labels from inputs. This enables ``mmengine.Evaluator`` to handle the data
    flow of different tasks through a unified interface.

    In addition, this wrapper also implements the ``evaluate`` method that
    parses metric results and print pretty table of metrics per class.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to return the computed  results of each
            class. Defaults to False.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        keep_results (bool): Whether to keep the results. When ``format_only``
            is True, ``keep_results`` must be True. If False, the result files
            will remove after compute the metric. Defaults to False.
        direct_compute (bool): Whether to compute metric on each inference
            iteration. Defaults to True.
        nproc (int): Number of processes for panoptic quality computing. It
            will be used when `direct_compute` is False. Defaults to 32.
            When ``nproc`` exceeds the number of cpu cores, the number of
            cpu cores is used.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        logger (Logger, optional): logger used to record messages. When set to
            ``None``, the default logger will be used.
            Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`.
    """
    default_prefix: Optional[str] = 'coco_panoptic'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 keep_results: bool = False,
                 nproc: int = 32,
                 direct_compute: bool = True,
                 backend_args: Optional[dict] = None,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:

        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`CityScapesMetric` is deprecated, use `dist_backend` '
                'instead.')

        logger = MMLogger.get_current_instance()

        super().__init__(
            ann_file=ann_file,
            seg_prefix=seg_prefix,
            classwise=classwise,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            keep_results=keep_results,
            nproc=nproc,
            direct_compute=direct_compute,
            backend_args=backend_args,
            logger=logger,
            dist_backend=dist_backend,
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
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                'jpg', 'png')
            # parse pred
            pred = dict()
            # shape (1, H, W) -> (H, W)
            sem_seg = data_sample['pred_panoptic_seg']['sem_seg'].cpu().numpy(
            )[0]
            pred['image_id'] = img_id
            pred['sem_seg'] = sem_seg
            pred['segm_file'] = segm_file
            predictions.append(pred)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['segm_file'] = segm_file
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            if self._coco_api is None:
                # get segments_info from dataset
                if 'segments_info' not in data_sample:
                    raise KeyError(
                        '`segments_info` is not in data_samples, please add '
                        '`segments_info` in PackDetInputs.meta_keys')
                gt['segments_info'] = data_sample['segments_info']
            groundtruths.append(gt)

        self.add(predictions, groundtruths)

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

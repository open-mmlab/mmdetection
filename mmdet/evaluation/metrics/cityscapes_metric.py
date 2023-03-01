# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
import warnings
from typing import Optional, Sequence

import numpy as np
from mmengine.logging import MMLogger, print_log
from mmeval import CityScapesDetection
from terminaltables import AsciiTable

from mmdet.registry import METRICS

try:
    import cityscapesscripts
except ImportError:
    cityscapesscripts = None


@METRICS.register_module()
class CityScapesMetric(CityScapesDetection):
    """A wrapper of :class:`mmeval.CityScapesDetection`.

    Args:
        outfile_prefix (str): The prefix of txt and png files. It is the
            saving path of txt and png file, e.g. "a/b/prefix".
        seg_prefix (str, optional): Path to the directory which contains the
            cityscapes instance segmentation masks. It's necessary when
            training and validation. It could be None when infer on test
            dataset. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        keep_results (bool): Whether to keep the results. When ``format_only``
            is True, ``keep_results`` must be True. Defaults to False.
        classwise (bool): Whether to return the computed  results of each
            class. Defaults to True.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'cityscapes'

    def __init__(self,
                 outfile_prefix: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 format_only: bool = False,
                 keep_results: bool = False,
                 classwise: bool = True,
                 keep_gt_json: bool = False,
                 prefix: Optional[str] = None,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:

        if cityscapesscripts is None:
            raise RuntimeError('Please run `pip install cityscapesscripts` to '
                               'install cityscapesscripts first.')

        collect_device = kwargs.pop('collect_device', None)
        if collect_device is not None:
            warnings.warn(
                'DeprecationWarning: The `collect_device` parameter of '
                '`CityScapesMetric` is deprecated, use `dist_backend` '
                'instead.')

        logger = MMLogger.get_current_instance()

        super().__init__(
            outfile_prefix=outfile_prefix,
            seg_prefix=seg_prefix,
            format_only=format_only,
            keep_results=keep_results,
            classwise=classwise,
            keep_gt_json=keep_gt_json,
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
            # parse pred
            pred = dict()
            pred_instances = data_sample['pred_instances']
            filename = data_sample['img_path']
            basename = osp.splitext(osp.basename(filename))[0]
            labels = pred_instances['labels'].cpu().numpy()
            masks = pred_instances['masks'].cpu().numpy().astype(np.uint8)
            if 'mask_scores' in pred_instances:
                # some detectors use different scores for bbox and mask
                mask_scores = pred_instances['mask_scores'].cpu().numpy()
            else:
                mask_scores = pred_instances['scores'].cpu().numpy()

            pred['labels'] = labels
            pred['masks'] = masks
            pred['mask_scores'] = mask_scores
            pred['basename'] = basename
            predictions.append(pred)

            # parse gt
            gt = dict()
            img_path = filename.replace('leftImg8bit.png',
                                        'gtFine_instanceIds.png')
            img_path = img_path.replace('leftImg8bit', 'gtFine')
            gt['file_name'] = img_path
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

        result = metric_results.pop('results_list', None)
        if result is None:
            return metric_results

        header = ['class', 'AP(%)', 'AP50(%)']
        table_title = ' Cityscapes Results'

        results_flatten = list(itertools.chain(*result))

        results_2d = itertools.zip_longest(
            *[results_flatten[i::3] for i in range(3)])
        table_data = [header]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data, title=table_title)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger='current')
        evaluate_results = {
            f'{self.prefix}/{k}(%)': round(float(v) * 100, 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

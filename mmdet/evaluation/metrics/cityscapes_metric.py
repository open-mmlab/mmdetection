# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.dist import is_main_process, master_only
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS

try:
    import cityscapesscripts
    from cityscapesscripts.evaluation import \
        evalInstanceLevelSemanticLabeling as CSEval
    from cityscapesscripts.helpers import labels as CSLabels
except ImportError:
    cityscapesscripts = None
    CSLabels = None
    CSEval = None


@METRICS.register_module()
class CityScapesMetric(BaseMetric):
    """CityScapes metric for instance segmentation.

    Args:
        outfile_prefix (str): The prefix of txt and png files. The txt and
            png file will be save in a directory whose path is
            "outfile_prefix.results/".
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
                 outfile_prefix: str,
                 seg_prefix: Optional[str] = None,
                 format_only: bool = False,
                 keep_results: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if cityscapesscripts is None:
            raise RuntimeError('Please run "pip install cityscapesscripts" to '
                               'install cityscapesscripts first.')

        assert outfile_prefix, 'outfile_prefix must be not None.'

        if format_only:
            assert keep_results, 'keep_results must be True when '
            'format_only is True'

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.format_only = format_only
        self.keep_results = keep_results
        self.seg_out_dir = osp.abspath(f'{outfile_prefix}.results')
        self.seg_prefix = seg_prefix

        if is_main_process():
            os.makedirs(self.seg_out_dir, exist_ok=True)

    @master_only
    def __del__(self) -> None:
        """Clean up."""
        if not self.keep_results:
            shutil.rmtree(self.seg_out_dir)

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
            # parse pred
            result = dict()
            pred = data_sample['pred_instances']
            filename = data_sample['img_path']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(self.seg_out_dir, basename + '_pred.txt')
            result['pred_txt'] = pred_txt
            labels = pred['labels'].cpu().numpy()
            masks = pred['masks'].cpu().numpy().astype(np.uint8)
            if 'mask_scores' in pred:
                # some detectors use different scores for bbox and mask
                mask_scores = pred['mask_scores'].cpu().numpy()
            else:
                mask_scores = pred['scores'].cpu().numpy()

            with open(pred_txt, 'w') as f:
                for i, (label, mask, mask_score) in enumerate(
                        zip(labels, masks, mask_scores)):
                    class_name = self.dataset_meta['CLASSES'][label]
                    class_id = CSLabels.name2label[class_name].id
                    png_filename = osp.join(
                        self.seg_out_dir, basename + f'_{i}_{class_name}.png')
                    mmcv.imwrite(mask, png_filename)
                    f.write(f'{osp.basename(png_filename)} '
                            f'{class_id} {mask_score}\n')

            # parse gt
            gt = dict()
            img_path = filename.replace('leftImg8bit.png',
                                        'gtFine_instanceIds.png')
            img_path = img_path.replace('leftImg8bit', 'gtFine')
            gt['file_name'] = osp.join(self.seg_prefix, img_path)

            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.format_only:
            logger.info(
                f'results are saved to {osp.dirname(self.seg_out_dir)}')
            return OrderedDict()
        logger.info('starts to compute metric')

        gts, preds = zip(*results)
        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = osp.join(self.seg_prefix, '../..')
        CSEval.args.predictionPath = self.seg_out_dir
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = osp.join(self.seg_out_dir,
                                               'gtInstances.json')

        groundTruthImgList = [gt['file_name'] for gt in gts]
        predictionImgList = [pred['pred_txt'] for pred in preds]
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']
        eval_results = OrderedDict()
        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']

        return eval_results

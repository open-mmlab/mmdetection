# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import OrderedDict
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS

try:
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa: E501
    import cityscapesscripts.helpers.labels as CSLabels

    from mmdet.evaluation.functional import evaluateImgLists
    HAS_CITYSCAPESAPI = True
except ImportError:
    HAS_CITYSCAPESAPI = False


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
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        dump_matches (bool): Whether dump matches.json file during evaluating.
            Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """
    default_prefix: Optional[str] = 'cityscapes'

    def __init__(self,
                 outfile_prefix: str,
                 seg_prefix: Optional[str] = None,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dump_matches: bool = False,
                 file_client_args: dict = None,
                 backend_args: dict = None) -> None:

        if not HAS_CITYSCAPESAPI:
            raise RuntimeError('Failed to import `cityscapesscripts`.'
                               'Please try to install official '
                               'cityscapesscripts by '
                               '"pip install cityscapesscripts"')
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.tmp_dir = None
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        else:
            assert seg_prefix is not None, '`seg_prefix` is necessary when '
            'computing the CityScapes metrics'

        if outfile_prefix is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(self.tmp_dir.name, 'results')
        else:
            # the directory to save predicted panoptic segmentation mask
            self.outfile_prefix = osp.join(outfile_prefix, 'results')  # type: ignore # yapf: disable # noqa: E501

        dir_name = osp.expanduser(self.outfile_prefix)

        if osp.exists(dir_name) and is_main_process():
            logger: MMLogger = MMLogger.get_current_instance()
            logger.info('remove previous results.')
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        self.seg_prefix = seg_prefix
        self.dump_matches = dump_matches

    def __del__(self) -> None:
        """Clean up the results if necessary."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

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
            pred_txt = osp.join(self.outfile_prefix, basename + '_pred.txt')
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
                    class_name = self.dataset_meta['classes'][label]
                    class_id = CSLabels.name2label[class_name].id
                    png_filename = osp.join(
                        self.outfile_prefix,
                        basename + f'_{i}_{class_name}.png')
                    mmcv.imwrite(mask, png_filename)
                    f.write(f'{osp.basename(png_filename)} '
                            f'{class_id} {mask_score}\n')

            # parse gt
            gt = dict()
            img_path = filename.replace('leftImg8bit.png',
                                        'gtFine_instanceIds.png')
            gt['file_name'] = img_path.replace('leftImg8bit', 'gtFine')

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
                f'results are saved to {osp.dirname(self.outfile_prefix)}')
            return OrderedDict()
        logger.info('starts to compute metric')

        gts, preds = zip(*results)
        # set global states in cityscapes evaluation API
        gt_instances_file = osp.join(self.outfile_prefix, 'gtInstances.json')  # type: ignore # yapf: disable # noqa: E501
        # split gt and prediction list
        gts, preds = zip(*results)
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = gt_instances_file

        groundTruthImgList = [gt['file_name'] for gt in gts]
        predictionImgList = [pred['pred_txt'] for pred in preds]
        CSEval_results = evaluateImgLists(
            predictionImgList,
            groundTruthImgList,
            CSEval.args,
            self.backend_args,
            dump_matches=self.dump_matches)['averages']

        eval_results = OrderedDict()
        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']

        return eval_results

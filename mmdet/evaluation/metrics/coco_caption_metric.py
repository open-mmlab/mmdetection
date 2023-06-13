# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import tempfile
from typing import List, Optional

from mmengine.evaluator import BaseMetric
from mmengine.utils import track_iter_progress
from pycocotools.coco import COCO

from mmdet.registry import METRICS

try:
    from pycocoevalcap.eval import COCOEvalCap
except ImportError:
    COCOEvalCap = None


@METRICS.register_module()
class COCOCaptionMetric(BaseMetric):
    """Coco Caption evaluation wrapper.

    Save the generated captions and transform into coco format.
    Calling COCO API for caption metrics.

    Args:
        ann_file (str): the path for the COCO format caption ground truth
            json file, load for evaluations.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """

    def __init__(self,
                 ann_file: str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        if COCOEvalCap is None:
            raise RuntimeError(
                'COCOEvalCap is not installed, please install it by: '
                'pip install pycocoevalcap')

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            result['caption'] = data_sample['pred_caption']
            result['image_id'] = int(data_sample['img_id'])

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.

        with tempfile.TemporaryDirectory() as temp_dir:

            eval_result_file = save_result(
                result=results,
                result_dir=temp_dir,
                filename='caption_pred',
                remove_duplicate='image_id',
            )

            coco_val = coco_caption_eval(eval_result_file, self.ann_file)

        return coco_val


def save_result(result, result_dir, filename, remove_duplicate=''):
    """Saving predictions as json file for evaluation."""
    # combine results from all processes
    if remove_duplicate:
        result_new = []
        id_list = []
        for res in track_iter_progress(result):
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new

    final_result_file_url = os.path.join(result_dir, '%s.json' % filename)
    print(f'result file saved to {final_result_file_url}')
    json.dump(result, open(final_result_file_url, 'w'))

    return final_result_file_url


def coco_caption_eval(results_file, ann_file):
    """Evaluation between gt json and prediction json files."""
    # create coco object and coco_result object
    coco = COCO(ann_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # make sure the image ids are the same
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # This will take some times at the first run
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval.eval

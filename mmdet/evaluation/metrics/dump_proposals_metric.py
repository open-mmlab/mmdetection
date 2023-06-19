# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Sequence

from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from mmdet.registry import METRICS


@METRICS.register_module()
class DumpProposals(BaseMetric):
    """Dump proposals pseudo metric.

    Args:
        output_dir (str): The root directory for ``proposals_file``.
            Defaults to ''.
        proposals_file (str): Proposals file path. Defaults to 'proposals.pkl'.
        num_max_proposals (int, optional): Maximum number of proposals to dump.
            If not specified, all proposals will be dumped.
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
    """

    default_prefix: Optional[str] = 'dump_proposals'

    def __init__(self,
                 output_dir: str = '',
                 proposals_file: str = 'proposals.pkl',
                 num_max_proposals: Optional[int] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_max_proposals = num_max_proposals
        # TODO: update after mmengine finish refactor fileio.
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        self.output_dir = output_dir
        assert proposals_file.endswith(('.pkl', '.pickle')), \
            'The output file must be a pkl file.'

        self.proposals_file = os.path.join(self.output_dir, proposals_file)
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            # `bboxes` is sorted by `scores`
            ranked_scores, rank_inds = pred['scores'].sort(descending=True)
            ranked_bboxes = pred['bboxes'][rank_inds, :]

            ranked_bboxes = ranked_bboxes.cpu().numpy()
            ranked_scores = ranked_scores.cpu().numpy()

            pred_instance = InstanceData()
            pred_instance.bboxes = ranked_bboxes
            pred_instance.scores = ranked_scores
            if self.num_max_proposals is not None:
                pred_instance = pred_instance[:self.num_max_proposals]

            img_path = data_sample['img_path']
            # `file_name` is the key to obtain the proposals from the
            # `proposals_list`.
            file_name = osp.join(
                osp.split(osp.split(img_path)[0])[-1],
                osp.split(img_path)[-1])
            result = {file_name: pred_instance}
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """Dump the processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: An empty dict.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        dump_results = {}
        for result in results:
            dump_results.update(result)
        dump(
            dump_results,
            file=self.proposals_file,
            backend_args=self.backend_args)
        logger.info(f'Results are saved at {self.proposals_file}')
        return {}

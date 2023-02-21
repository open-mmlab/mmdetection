# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence, Union

from mmengine.logging import print_log
from mmeval.metrics import ProposalRecall as _ProposalRecall
from terminaltables import AsciiTable

from mmdet.registry import METRICS


@METRICS.register_module()
class ProposalRecallMetric(_ProposalRecall):
    """Proposals recall evaluation metric.

    The speed of calculating recall is faster than COCO Detection metric.

    Args:
        iou_thrs (float | List[float], optional): IoU thresholds.
            If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (1, 10, 100, 1000).
            Note: it defaults to (100, 300, 1000) in MMDet 2.x.
        use_legacy_coordinate (bool): Whether to use coordinate
            system in mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Please set `True` when using VOCDataset. Defaults to False.
        nproc (int): Processes used for computing TP and FP. If nproc
            is less than or equal to 1, multiprocessing will not be used.
            Defaults to 4.
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 proposal_nums: Union[int, Sequence[int]] = (100, 300, 1000),
                 use_legacy_coordinate: bool = False,
                 nproc: int = 4,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            iou_thrs=iou_thrs,
            proposal_nums=proposal_nums,
            use_legacy_coordinate=use_legacy_coordinate,
            nproc=nproc,
            dist_backend=dist_backend,
            **kwargs)

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. Parse predictions
        and ground truths from ``data_samples`` and invoke ``self.add``.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy())
            groundtruths.append(ann)

            pred = data_sample['pred_instances']
            pred['bboxes'] = pred['bboxes'].cpu().numpy()
            pred['scores'] = pred['scores'].cpu().numpy()
            predictions.append(pred)

        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        import time
        t1 = time.time()
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        t2 = time.time()
        print(t2 - t1)

        result = metric_results.pop('proposal_result')
        headers = [''] + [
            f'AR_{iou_thr * 100:.0f}' for iou_thr in self.iou_thrs
        ] + ['AR']
        table_data = [headers]
        for i in range(len(self.proposal_nums)):
            row = [f'{self.proposal_nums[i]}'] + \
                  [f'{100 * round(val, 4):.2f}' for val in result[i].tolist()]
            table_data.append(row)
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger='current')

        evaluate_results = {
            k: round(float(v), 4)
            for k, v in metric_results.items()
        }
        return evaluate_results

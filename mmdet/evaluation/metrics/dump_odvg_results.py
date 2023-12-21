# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence

from mmcv.ops import batched_nms
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log

from mmdet.registry import METRICS

try:
    import jsonlines
except ImportError:
    jsonlines = None


@METRICS.register_module()
class DumpODVGResults(BaseMetric):
    default_prefix: Optional[str] = 'pl_odvg'

    def __init__(self,
                 outfile_path,
                 img_prefix: str,
                 score_thr: float = 0.1,
                 collect_device: str = 'cpu',
                 nms_thr: float = 0.5,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.outfile_path = outfile_path
        self.score_thr = score_thr
        self.img_prefix = img_prefix
        self.nms_thr = nms_thr

        if jsonlines is None:
            raise ImportError('Please run "pip install jsonlines" to install '
                              'this package.')

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = {}

            filename = data_sample['img_path']
            filename = filename.replace(self.img_prefix, '')
            if filename.startswith('/'):
                filename = filename[1:]
            result['filename'] = filename

            height = data_sample['ori_shape'][0]
            width = data_sample['ori_shape'][1]
            result['height'] = height
            result['width'] = width

            pred_instances = data_sample['pred_instances']

            bboxes = pred_instances['bboxes'].cpu()
            scores = pred_instances['scores'].cpu()
            labels = pred_instances['labels'].cpu()

            bboxes = bboxes[scores > self.score_thr]
            labels = labels[scores > self.score_thr]
            scores = scores[scores > self.score_thr]

            if 'tokens_positive' in data_sample:
                task = 'vg'
            else:
                task = 'od'

            if task == 'od':
                classes_name = data_sample['text']
                result['detection'] = {}

                if len(bboxes) > 0:
                    det_bboxes, keep = batched_nms(
                        bboxes, scores, labels,
                        dict(type='nms', iou_threshold=self.nms_thr))
                    _scores = det_bboxes[:, -1]
                    _bboxes = det_bboxes[:, :-1]
                    _labels = labels[keep]

                    instances = []
                    _bboxes = _bboxes.numpy().tolist()
                    _scores = _scores.numpy().tolist()
                    _labels = _labels.numpy().tolist()
                    for bbox, score, label in zip(_bboxes, _scores, _labels):
                        round_bbox = [round(b, 2) for b in bbox]
                        round_score = round(score, 2)
                        instances.append({
                            'bbox': round_bbox,
                            'score': round_score,
                            'label': label,
                            'category': classes_name[label]
                        })
                    result['detection']['instances'] = instances
                else:
                    result['detection']['instances'] = []
                self.results.append(result)
            else:
                caption = data_sample['text']
                result['grounding'] = {}
                result['grounding']['caption'] = caption

                tokens_positive = data_sample['tokens_positive']

                region_list = []
                for label, positive in enumerate(tokens_positive):
                    phrase = [caption[pos[0]:pos[1]] for pos in positive]

                    _bboxes = bboxes[labels == label]
                    _scores = scores[labels == label]
                    det_bboxes, _ = batched_nms(
                        _bboxes,
                        _scores,
                        None,
                        dict(type='nms', iou_threshold=self.nms_thr),
                        class_agnostic=True)
                    _scores = det_bboxes[:, -1].numpy().tolist()
                    _bboxes = det_bboxes[:, :-1].numpy().tolist()

                    round_bboxes = []
                    for bbox in _bboxes:
                        round_bboxes.append([round(b, 2) for b in bbox])
                    _scores = [[round(s, 2) for s in _scores]]
                    region = {
                        'phrase': phrase,
                        'bbox': round_bboxes,
                        'score': _scores,
                        'tokens_positive': positive
                    }
                    region_list.append(region)
                result['grounding']['regions'] = region_list
                self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        with jsonlines.open(self.outfile_path, mode='w') as writer:
            writer.write_all(results)
        print_log(
            f'Results has been saved to {self.outfile_path}.',
            logger='current')
        return {}

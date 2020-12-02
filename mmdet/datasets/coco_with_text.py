import collections
import copy
import itertools
import logging
import os.path as osp
import string
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.core import text_eval
from .builder import DATASETS
from .coco import CocoDataset, ConcatenatedCocoDataset, get_polygon


@DATASETS.register_module()
class CocoWithTextDataset(CocoDataset):

    CLASSES = ('text')

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['text_fields'] = []

    def __init__(self, alphabet='  ' + string.ascii_lowercase + string.digits, max_texts_num=0, *args, **kwargs):
        self.max_texts_num = max_texts_num
        super().__init__(*args, **kwargs)
        self.alphabet = alphabet
        self.max_text_len = 33
        self.EOS = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = collections.Counter(_['image_id'] for _ in self.coco.anns.values())
        if self.max_texts_num > 0:
            ids_with_ann = {k for k, v in ids_with_ann.items() if v <= self.max_texts_num}
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_texts = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            text = ann['text']['transcription'] if ann['text']['legible'] else ''
            text = text.lower()
            assert not ann.get('iscrowd', False) == ann['text']['legible']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if self.min_size is not None:
                if w < self.min_size or h < self.min_size:
                    continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

                if ' ' in text:
                    text = []
                else:
                    text = [self.alphabet.find(l) for l in text]
                    if -1 in text:
                        text = []
                    else:
                        text.append(self.EOS)
                text = np.array(text)
                gt_texts.append(text)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.empty((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_texts:
            gt_texts = np.array(gt_texts)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.empty((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            texts=gt_texts)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 score_thr=-1):

        metrics = metric if isinstance(metric, list) else [metric]

        computed_metrics = ['word_spotting']
        removed_metrics = []

        for computed_metric in computed_metrics:
            if computed_metric in metrics:
                metrics.remove(computed_metric)
                removed_metrics.append(computed_metric)
        eval_results = super().evaluate(results, metrics, logger, jsonfile_prefix, classwise, proposal_nums, iou_thrs, score_thr)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        for metric in removed_metrics:
            cocoGt = copy.deepcopy(self.coco)
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            metric_type = 'bbox' if metric in ['word_spotting'] else metric
            if metric_type not in result_files:
                raise KeyError(f'{metric_type} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric_type])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric in {'word_spotting'} else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            if metric in ['word_spotting']:
                predictions = []
                for res in results:
                    boxes = res[0][0]
                    segms = res[1][0]
                    texts = res[2]

                    per_image_predictions = []

                    for bbox, segm, text in zip(boxes, segms, texts):
                        if text or metric == 'f1':
                            text = text.upper()
                            contour = get_polygon(segm, bbox)
                            per_image_predictions.append({
                                'segmentation': contour,
                                'score': 1.0,
                                'text': {
                                    'transcription': text
                                }
                            })

                    predictions.append(per_image_predictions)

                gt_annotations = cocoEval.cocoGt.imgToAnns
                recall, precision, hmean, _ = text_eval(
                    predictions, gt_annotations, score_thr,
                    show_recall_graph=False,
                    use_transcriptions=True)
                print('Text detection recall={:.4f} precision={:.4f} hmean={:.4f}'.
                      format(recall, precision, hmean))
                eval_results[metric + '/hmean'] = float(f'{hmean:.3f}')
                eval_results[metric + '/precision'] = float(f'{precision:.3f}')
                eval_results[metric + '/recall'] = float(f'{recall:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


@DATASETS.register_module()
class ConcatenatedCocoWithTextDataset(CocoWithTextDataset, ConcatenatedCocoDataset):
    def __init__(self, concatenated_dataset):
        ConcatenatedCocoDataset.__init__(self, concatenated_dataset)
        self.max_texts_num = concatenated_dataset.datasets[0].max_texts_num
        self.alphabet = concatenated_dataset.datasets[0].alphabet
        self.max_text_len = concatenated_dataset.datasets[0].max_text_len
        self.EOS = concatenated_dataset.datasets[0].EOS

        assert all(self.max_texts_num == x.max_texts_num for x in concatenated_dataset.datasets)
        assert all(self.alphabet == x.alphabet for x in concatenated_dataset.datasets)
        assert all(self.max_text_len == x.max_text_len for x in concatenated_dataset.datasets)
        assert all(self.EOS == x.EOS for x in concatenated_dataset.datasets)

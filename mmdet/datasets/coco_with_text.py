import collections
import copy
import logging
import os
import string
import subprocess
import tempfile

import editdistance
import numpy as np
from mmcv.utils import print_log
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mmdet.core import text_eval
from mmdet.core.evaluation.text_evaluation import strip
from .builder import DATASETS
from .coco import CocoDataset, ConcatenatedCocoDataset, get_polygon
from .weighted_editdistance import weighted_edit_distance


def find_in_lexicon(sequence, lexicon, lexicon_mapping, char_distrib):
    sequence = sequence.upper()
    distances = [editdistance.eval(sequence, word.upper()) for word in lexicon]
    argmin = np.argmin(distances)
    distance = distances[argmin]
    if char_distrib is not None:
        small_lexicon = [word for i, word in enumerate(lexicon) if distances[i] <= distance + 2]
        distances = [weighted_edit_distance(sequence, word.upper(), char_distrib) for word in small_lexicon]
        argmin = np.argmin(distances)
        distance = distances[argmin]
        word = small_lexicon[argmin]
    else:
        word = lexicon[argmin]
    if lexicon_mapping:
        word = lexicon_mapping[word]
    return word, distance

@DATASETS.register_module()
class CocoWithTextDataset(CocoDataset):

    CLASSES = ('text')

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['text_fields'] = []

    def __init__(self, alphabet='  ' + string.ascii_lowercase + string.digits,
                 max_texts_num=0, *args, **kwargs):
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
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            for text_key in ('text', 'attributes'):
                if text_key in ann:
                    break
            legible = ann[text_key]['legible']
            text = ann[text_key]['transcription'] if legible else ''
            text = text.lower()
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
            if ann.get('iscrowd', False) or not legible:
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

    @staticmethod
    def _parse_metric(metric):
        metric = metric.split('@')
        metric_params = {
            'lexicon': None,
            'lexicon_mapping': None,
            'det_thr' : 0.0,
            'rec_thr' : 0.0,
            'dataset': 'icdar15'
        }
        if len(metric) == 2:
            params = {k:v for k,v in [kv.split('=') for kv in metric[1].split(',')]}
            metric_params.update(params)
            metric_params['det_thr'] = float(metric_params['det_thr'])
            metric_params['rec_thr'] = float(metric_params['rec_thr'])
        metric = metric[0]

        assert metric_params['dataset'] in ('icdar15', 'totaltext')
        allowed_keys = {'lexicon', 'lexicon_mapping', 'det_thr', 'rec_thr', 'dataset'}
        assert all((k in allowed_keys) for k in metric_params.keys())

        return metric, metric_params

    def _dump_predictions(self, predictions, dataset_format):
        tempdir = tempfile.mkdtemp()
        assert dataset_format in ('icdar15', 'totaltext')
        suffix = 'res_' if dataset_format == 'icdar15' else 'gt_'
        for i, per_image_predictions in tqdm(enumerate(predictions)):
            image_basename = os.path.basename(self.data_infos[i]['filename'])
            dest = suffix + image_basename[:-3] + 'txt'
            dest = f'{tempdir}/{dest}'
            with open(dest, 'w') as write_file:
                for prediction in per_image_predictions:
                    contour = prediction['segmentation']
                    text = prediction['text']['transcription']
                    s = ','.join([str(int(x)) for x in contour]) + ',' + text
                    write_file.write(s + '\n')

        subprocess.run(f'cd {tempdir}; zip -q pr.zip *', check=True, shell=True)
        print(f'Results archived to {tempdir}/pr.zip')

    @staticmethod
    def _filter_predictions(predictions, det_thr, rec_thr):
        filtered_predictions = []
        for per_image_predictions in predictions:
            filtered_per_image_predictions = [p for p in per_image_predictions if p['score'] >= det_thr]
            filtered_per_image_predictions = [p for p in filtered_per_image_predictions if p['text']['score'] >= rec_thr]
            filtered_predictions.append(filtered_per_image_predictions)
        return filtered_predictions

    @staticmethod
    def _read_lexicon(path):
        if not path:
            return []
        with open(path) as read_file:
            lexicon = [line.strip() for line in read_file]
        return lexicon

    @staticmethod
    def _read_lexicon_mapping(path):
        if not path:
            return {}
        with open(path) as read_file:
            lexicon_mapping = [line.strip().split(' ') for line in read_file]
            lexicon_mapping = {pair[0].upper(): ' '.join(pair[1:]) for pair in lexicon_mapping}
        return lexicon_mapping

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 score_thr=-1):

        metrics = list(metric) if isinstance(metric, list) else [metric]

        computed_metrics = ['word_spotting', 'e2e_recognition']
        removed_metrics = []

        for metric in metrics:
            if any(metric.startswith(computed_metric) for computed_metric in computed_metrics):
                removed_metrics.append(metric)

        metrics = [metric for metric in metrics if metric not in removed_metrics]

        for metric in removed_metrics:
            if metric.split('@')[0] not in computed_metrics:
                raise RuntimeError(f'Unknonwn metric: {metric}')

        eval_results = super().evaluate(results, metrics, logger, jsonfile_prefix, classwise, proposal_nums, iou_thrs, score_thr)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        for metric in removed_metrics:
            cocoGt = copy.deepcopy(self.coco)
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            metric, metric_params = self._parse_metric(metric)

            metric_type = 'bbox'
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

            iou_type = 'bbox'
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            lexicon_path = metric_params['lexicon']
            lexicon = self._read_lexicon(lexicon_path)

            lexicon_pairs_path = metric_params['lexicon_mapping']
            lexicon_mapping = self._read_lexicon_mapping(lexicon_pairs_path)

            predictions = []
            for res in tqdm(results):
                boxes = res[0][0]
                segms = res[1][0]
                texts, text_confidences, character_distributions = res[2]
                per_image_predictions = []

                for bbox, segm, text, text_conf, char_distrib in zip(boxes, segms, texts, text_confidences, character_distributions):
                    if text:
                        text = text.upper()
                        if lexicon:
                            text, _ = find_in_lexicon(text, lexicon, lexicon_mapping, char_distrib)
                            if metric == 'word_spotting':
                                text = strip(text)
                        contour, conf = get_polygon(segm, bbox, metric_params['dataset'] == 'icdar15')
                        per_image_predictions.append({
                            'segmentation': [int(x) for x in contour],
                            'score': conf,
                            'text': {
                                'transcription': text,
                                'score' : text_conf
                            }
                        })
                predictions.append(per_image_predictions)
            gt_annotations = cocoEval.cocoGt.imgToAnns
            filtered_predictions = self._filter_predictions(
                predictions, metric_params['det_thr'], metric_params['rec_thr']
            )
            self._dump_predictions(filtered_predictions, metric_params['dataset'])
            recall, precision, hmean, _ = text_eval(
                filtered_predictions, gt_annotations, score_thr,
                show_recall_graph=False,
                use_transcriptions=True,
                word_spotting=metric.startswith('word_spotting'))

            print(f'Text detection recall={recall} precision={precision} hmean={hmean}')
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

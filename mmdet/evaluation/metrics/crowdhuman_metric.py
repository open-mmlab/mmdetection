# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os.path as osp
import tempfile
from collections import OrderedDict
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tqdm import tqdm

from mmdet.registry import METRICS

PERSON_CLASSES = ['background', 'person']


@METRICS.register_module()
class CrowdHumanMetric(BaseMetric):

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'AP',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        # crowdhuman evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MR', 'AP', 'JI']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric should be one of 'MR', 'AP', 'JI',"
                               f'but got {metric}.')

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'
        self.outfile_prefix = outfile_prefix

    @staticmethod
    def results2json(results: Sequence[dict], outfile_prefix: str) -> str:
        """Dump the detection results to a json file."""
        result_file_path = f'{outfile_prefix}.json'
        with open(result_file_path, 'w') as outfile:
            for i, result in enumerate(results):
                ann, pred = result
                dump_dict = dict()
                dump_dict['ID'] = ann['ID']
                dump_dict['width'] = ann['width']
                dump_dict['height'] = ann['height']
                dtboxes = []
                bboxes = pred.tolist()
                for _, single_bbox in enumerate(bboxes):
                    temp_dict = dict()
                    x1, y1, x2, y2, score = single_bbox
                    temp_dict['box'] = [x1, y1, x2 - x1, y2 - y1]
                    temp_dict['score'] = score
                    temp_dict['tag'] = 1
                    dtboxes.append(temp_dict)
                dump_dict['dtboxes'] = dtboxes
                outfile.write(json.dumps(dump_dict) + '\n')

        return result_file_path

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data, pred in zip(data_batch, predictions):
            gt = copy.deepcopy(data['data_sample'])
            ann = dict()

            ann['ID'] = gt['img_id']
            ann['width'] = gt['ori_shape'][1]
            ann['height'] = gt['ori_shape'][0]

            pred_bboxes = pred['pred_instances']['bboxes'].cpu().numpy()
            pred_scores = pred['pred_instances']['scores'].cpu().numpy()

            pred_bbox_scores = np.hstack(
                [pred_bboxes, pred_scores.reshape((-1, 1))])

            self.results.append((ann, pred_bbox_scores))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'result')
        else:
            outfile_prefix = self.outfile_prefix

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(results, outfile_prefix)
        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        database = Database(self.ann_file, result_files, 'box', None, 0)
        database.compare()
        # return dict(bbox_mAP=AP, mMR=mMR, recall=recall)
        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')
            if metric == 'AP':
                AP, recall, _ = database.eval_AP()
                eval_results['mAP'] = float(f'{round(AP, 4)}')
                eval_results['recall'] = float(f'{round(recall, 4)}')
            if metric == 'MR':
                mMR, _ = database.eval_MR()
                eval_results['mMR'] = float(f'{round(mMR, 4)}')
            if metric == 'JI':
                _, JI = database.eval_JI()
                eval_results['JI'] = float(f'{round(JI, 4)}')
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results


class Image(object):
    """Data structure for evaluation of CrowdHuman.

    Args:
        mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
    """

    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def load(self, record, body_key, head_key, class_names, gt_flag):
        """Loading information for evaluation.

        Args:
            record (dict): Label information or test results.
                The format might look something like this:
                {
                    'ID': '273271,c9db000d5146c15',
                    'gtboxes': [
                        {'fbox': [72, 202, 163, 503], 'tag': 'person', ...},
                        {'fbox': [199, 180, 144, 499], 'tag': 'person', ...},
                        ...
                    ]
                }
                or:
                {
                    'ID': '273271,c9db000d5146c15',
                    'width': 800,
                    'height': 1067,
                    'dtboxes': [
                        {
                            'box': [306.22, 205.95, 164.05, 394.04],
                            'score': 0.99,
                            'tag': 1
                        },
                        {
                            'box': [403.60, 178.66, 157.15, 421.33],
                            'score': 0.99,
                            'tag': 1
                        },
                        ...
                    ]
                }
            body_key (str): key of detection body box.
                Valid when loading detection results and self.eval_mode!=1.
            head_key (str): key of detection head box.
                Valid when loading detection results and self.eval_mode!=0.
            class_names (list[str]):class names of data set.
                Defaults to ['background', 'person'].
            gt_flag (bool): Indicate whether record is ground truth
                or predicting the outcome.
        """
        if 'ID' in record and self.ID is None:
            self.ID = record['ID']
        if 'width' in record and self._width is None:
            self._width = record['width']
        if 'height' in record and self._height is None:
            self._height = record['height']
        if gt_flag:
            self._gtNum = len(record['gtboxes'])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes',
                                                      class_names)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array([
                    body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                    for i in range(len(body_bbox))
                ])
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack((body_bbox[:, :-1], head_bbox[:, :-1],
                                          gt_tag.reshape(-1, 1)))
            else:
                raise Exception('Unknown evaluation mode!')
        if not gt_flag:
            self._dtNum = len(record['dtboxes'])
            if self.eval_mode == 0:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key,
                                                   'score')
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key,
                                                   'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key,
                                                   'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """Match the detection results with the ground_truth by Caltech
        matching strategy.

        Args:
            thres (float): IOU threshold.

        Returns:
            scorelist: a list of tuples (dtbox, imageID),
                in the descending sort of dtbox.score.
        """
        if self.dtboxes is None or self.gtboxes is None:
            return list()

        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes, True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes, False)
        else:
            return list()

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech
        matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort
        of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:, None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:, None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    @staticmethod
    def box_overlap_opr(dboxes: np.ndarray, gboxes: np.ndarray, if_iou):
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

        iw = (
            np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2]) -
            np.maximum(dtboxes[:, :, 0], gtboxes[:, :, 0]))
        ih = (
            np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3]) -
            np.maximum(dtboxes[:, :, 1], gtboxes[:, :, 1]))
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:, :, 2] - dtboxes[:, :, 0]) * (
            dtboxes[:, :, 3] - dtboxes[:, :, 1])
        if if_iou:
            gtarea = (gtboxes[:, :, 2] - gtboxes[:, :, 0]) * (
                gtboxes[:, :, 3] - gtboxes[:, :, 1])
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):
        """Make sure boxes are within the image range."""

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dtboxes.shape[-1] >= 4
        assert self.gtboxes.shape[-1] >= 4
        assert self._width is not None and self._height is not None
        if self.eval_mode == 2:
            self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4],
                                                 self._height, self._width)
            self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4],
                                                 self._height, self._width)
            self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8],
                                                  self._height, self._width)
            self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8],
                                                  self._height, self._width)
        else:
            self.dtboxes = _clip_boundary(self.dtboxes, self._height,
                                          self._width)
            self.gtboxes = _clip_boundary(self.gtboxes, self._height,
                                          self._width)

    @staticmethod
    def load_gt_boxes(dict_input, key_name, class_names):
        """load ground_truth and transform [x, y, w, h] to [x1, y1, x2, y2]"""
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = copy.deepcopy(body_tag)
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            body_bbox.append(np.hstack((rb['fbox'], body_tag)))
        head_bbox = np.array(head_bbox)
        head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    @staticmethod
    def load_det_boxes(dict_input, key_name, key_box, key_score, key_tag=None):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_score], rb[key_tag]))
                    for rb in dict_input[key_name]
                ])
            else:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_score]))
                    for rb in dict_input[key_name]
                ])
        else:
            if key_tag:
                bboxes = np.vstack([
                    np.hstack((rb[key_box], rb[key_tag]))
                    for rb in dict_input[key_name]
                ])
            else:
                bboxes = np.vstack(
                    [rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC
        matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending
        sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist


class Database(object):
    """Loading information and evaluation.

    Args:
        gt_path (str): Path of an annotations file.
        dt_path (str): Path of detection results.
        body_key (str, None): key of detection body box.
            Valid when loading detection results and mode!=1.
        head_key (str, None): key of detection head box.
            Valid when loading detection results and mode!=0.
        mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
    Examples:
        >>> database = Database(
        >>>     '/data/YuYoujiang/dataset/CrowdHuman/annotation_val.odgt',
        >>>     '/data/YuYoujiang/CrowdDet/mmdetection/tools/result_meg.json',
        >>>     'box', None, 0)
        >>> database.compare()
        >>> eval_results = OrderedDict()
        >>> AP, recall, _ = database.eval_AP()
        >>> eval_results['mAP'] = float(f'{round(AP, 3)}')
        >>> eval_results['recall'] = float(f'{round(recall, 3)}')
        >>> mMR, _ = database.eval_MR()
        >>> eval_results['mMR'] = float(f'{round(mMR, 3)}')
        >>> print(eval_results)
    """

    def __init__(self,
                 gt_path=None,
                 dt_path=None,
                 body_key=None,
                 head_key=None,
                 mode=0):

        self.images = dict()
        self.eval_mode = mode
        self.loadData(gt_path, body_key, head_key, if_gt=True)
        self.loadData(dt_path, body_key, head_key, if_gt=False)

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None

    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True):
        """Load data from annotations file or detection results. Notice the
        order in which this method is called： Need to load annotation file
        before loading test results.

        Args:
            fpath (str): Path of an annotations file or detection results.
            body_key (str): key of detection full box. Defaults to None.
            head_key (str): key of detection head box. Defaults to None.
            if_gt (bool): Select the mode of evaluate. Valid mode include
                0(just body box), 1(just head box) and 2(both of them).
                Defaults to True.
        """
        assert osp.isfile(fpath), fpath + ' does not exist!'
        with open(fpath, 'r') as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        if if_gt:
            for record in records:
                self.images[record['ID']] = Image(self.eval_mode)
                self.images[record['ID']].load(record, body_key, head_key,
                                               PERSON_CLASSES, True)
        else:
            for record in records:
                self.images[record['ID']].load(record, body_key, head_key,
                                               PERSON_CLASSES, False)
                self.images[record['ID']].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """Match the detection results with the ground_truth.

        Args:
            thres (float): The processed results of each batch.
            matching(str): Matching rules. Valid rules include
                VOC and None(caltech). Defaults to None.
        Returns:
            list[tuple[ndarray, int, str]]: Matching result.
                The detection bboxes and scores.
                The label of bbox.
                Image ID.
        """
        assert matching is None or matching == 'VOC', matching
        scorelist = list()
        for ID in self.images:
            if matching == 'VOC':
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self.scorelist = scorelist

    def eval_MR(self, ref='CALTECH_-2'):
        """Evaluate by Caltech-style log-average miss rate.

        Args:
            ref (str): Different ways of calculating. Valid ref include
            CALTECH_-2 and CALTECH_-4. Defaults to CALTECH_-2.

        Returns:
            float: miss rate results.
        """

        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst) - 1

        assert ref == 'CALTECH_-2' or ref == 'CALTECH_-4', ref
        if ref == 'CALTECH_-2':
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in
            # P.Dollar's paper
            ref = [
                0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162,
                0.5623, 1.000
            ]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in
            # S.Zhang's paper
            ref = [
                0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000,
                0.3162, 1.000
            ]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list()
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = (self._gtNum - self._ignNum) - tp
            recall = tp / (tp + fn)
            missrate = 1.0 - recall
            fppi = fp / self._imageNum
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """Evaluate by average precision."""

        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i - 1] + precision[i]) / 2
                delta_w = recall[i] - recall[i - 1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list()
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / total_images)

        AP = _calculate_map(rpX, rpY)
        return AP, recall, (rpX, rpY, thr, fpn, recalln, fppi)

    def eval_JI(self, nr_procs=10):
        """Evaluate by JI."""
        import math
        res_line = []
        res_JI = []
        for i in range(10):
            score_thr = 1e-1 * i
            total = len(self.images)
            stride = math.ceil(total / nr_procs)
            result_queue = Queue(10000)
            results, procs = [], []
            records = list(self.images.items())
            for i in range(nr_procs):
                start = i * stride
                end = np.min([start + stride, total])
                sample_data = dict(records[start:end])
                p = Process(
                    target=compute_JI_with_ignore,
                    args=(result_queue, sample_data, score_thr))
                p.start()
                procs.append(p)
            tqdm.monitor_interval = 0
            pbar = tqdm(total=total, leave=False, ascii=True)
            for i in range(total):
                t = result_queue.get()
                results.append(t)
                pbar.update(1)
            for p in procs:
                p.join()
            pbar.close()
            line, mean_ratio = gather(results)
            line = 'score_thr:{:.1f}, {}'.format(score_thr, line)
            # print(line)
            res_line.append(line)
            res_JI.append(mean_ratio)
        return res_line, max(res_JI)


def gather(results):
    assert len(results)
    img_num = 0
    for result in results:
        if result['n'] != 0 or result['m'] != 0:
            img_num += 1
    mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
    valids = np.sum([rb['k'] for rb in results])
    total = np.sum([rb['n'] for rb in results])
    gtn = np.sum([rb['m'] for rb in results])
    line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}' \
        .format(mean_ratio, valids, total, gtn)
    return line, mean_ratio


def compute_JI_with_ignore(result_queue, images, score_thr, bm_thresh=0.5):
    for ID, record in images.items():
        gt_boxes = record.gtboxes
        dt_boxes = record.dtboxes
        keep = dt_boxes[:, -1] > score_thr
        dt_boxes = dt_boxes[keep][:, :-1]

        gt_tag = np.array(gt_boxes[:, -1] != -1)
        matches = compute_matching(dt_boxes, gt_boxes[gt_tag, :4], bm_thresh)
        # get the unmatched_indices
        matched_indices = np.array([j for (j, _) in matches])
        unmatched_indices = list(
            set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
        num_ignore_dt = get_ignores(dt_boxes[unmatched_indices],
                                    gt_boxes[~gt_tag, :4], bm_thresh)
        matched_indices = np.array([j for (_, j) in matches])
        unmatched_indices = list(
            set(np.arange(gt_boxes[gt_tag].shape[0])) - set(matched_indices))
        num_ignore_gt = get_ignores(gt_boxes[gt_tag][unmatched_indices],
                                    gt_boxes[~gt_tag, :4], bm_thresh)
        # compute results
        eps = 1e-6
        k = len(matches)
        m = gt_tag.sum() - num_ignore_gt
        n = dt_boxes.shape[0] - num_ignore_dt
        ratio = k / (m + n - k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover
        result_dict = dict(
            ratio=ratio,
            recall=recall,
            cover=cover,
            noise=noise,
            k=k,
            m=m,
            n=n)
        result_queue.put_nowait(result_dict)


def compute_matching(dt_boxes, gt_boxes, bm_thr):
    assert dt_boxes.shape[-1] > 3 and gt_boxes.shape[-1] > 3
    if dt_boxes.shape[0] < 1 or gt_boxes.shape[0] < 1:
        return list()
    N = dt_boxes.shape[0]
    ious = compute_iou_matrix(dt_boxes, gt_boxes)
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i - 1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches


def compute_iou_matrix(dboxes: np.ndarray, gboxes: np.ndarray):
    eps = 1e-6
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

    iw = np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2]) - np.maximum(
        dtboxes[:, :, 0], gtboxes[:, :, 0])
    ih = np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3]) - np.maximum(
        dtboxes[:, :, 1], gtboxes[:, :, 1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:, :, 2] - dtboxes[:, :, 0]) * (
        dtboxes[:, :, 3] - dtboxes[:, :, 1])
    gtarea = (gtboxes[:, :, 2] - gtboxes[:, :, 0]) * (
        gtboxes[:, :, 3] - gtboxes[:, :, 1])
    ious = inter / (dtarea + gtarea - inter + eps)
    return ious


def get_ignores(dt_boxes, gt_boxes, bm_thr):
    if gt_boxes.size:
        ioas = compute_ioa_matrix(dt_boxes, gt_boxes)
        ioas = np.max(ioas, axis=1)
        rows = np.where(ioas > bm_thr)[0]
        return len(rows)
    else:
        return 0


def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):
    eps = 1e-6
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

    iw = np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2]) - np.maximum(
        dtboxes[:, :, 0], gtboxes[:, :, 0])
    ih = np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3]) - np.maximum(
        dtboxes[:, :, 1], gtboxes[:, :, 1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:, :, 2] - dtboxes[:, :, 0], 0) * np.maximum(
        dtboxes[:, :, 3] - dtboxes[:, :, 1], 0)
    ioas = inter / (dtarea + eps)
    return ioas


def maxWeightMatching(edges, maxcardinality=False):
    if not edges:
        return []

    # Count vertices.
    nedge = len(edges)
    nvertex = 0
    for (i, j, w) in edges:
        assert i >= 0 and j >= 0 and i != j
        if i >= nvertex:
            nvertex = i + 1
        if j >= nvertex:
            nvertex = j + 1

    # Find the maximum edge weight.
    maxweight = max(0, max([wt for (i, j, wt) in edges]))

    # If p is an edge endpoint,
    # endpoint[p] is the vertex to which endpoint p is attached.
    # Not modified by the algorithm.
    endpoint = [edges[p // 2][p % 2] for p in range(2 * nedge)]

    # If v is a vertex,
    # neighbend[v] is the list of remote endpoints of the edges attached to v.
    # Not modified by the algorithm.
    neighbend = [[] for i in range(nvertex)]
    for k in range(len(edges)):
        (i, j, w) = edges[k]
        neighbend[i].append(2 * k + 1)
        neighbend[j].append(2 * k)

    # If v is a vertex,
    # mate[v] is the remote endpoint of its matched edge, or -1 if it is single
    # (i.e. endpoint[mate[v]] is v's partner vertex).
    # Initially all vertices are single; updated during augmentation.
    mate = nvertex * [-1]

    # If b is a top-level blossom,
    # label[b] is 0 if b is unlabeled (free);
    #             1 if b is an S-vertex/blossom;
    #             2 if b is a T-vertex/blossom.
    # The label of a vertex is found by looking at the label of its
    # top-level containing blossom.
    # If v is a vertex inside a T-blossom,
    # label[v] is 2 iff v is reachable from an S-vertex outside the blossom.
    # Labels are assigned during a stage and reset after each augmentation.
    label = (2 * nvertex) * [0]

    # If b is a labeled top-level blossom,
    # labelend[b] is the remote endpoint of the edge through which b obtained
    # its label, or -1 if b's base vertex is single.
    # If v is a vertex inside a T-blossom and label[v] == 2,
    # labelend[v] is the remote endpoint of the edge through which v is
    # reachable from outside the blossom.
    labelend = (2 * nvertex) * [-1]

    # If v is a vertex,
    # inblossom[v] is the top-level blossom to which v belongs.
    # If v is a top-level vertex, v is itself a blossom (a trivial blossom)
    # and inblossom[v] == v.
    # Initially all vertices are top-level trivial blossoms.
    inblossom = list(range(nvertex))

    # If b is a sub-blossom,
    # blossomparent[b] is its immediate parent (sub-)blossom.
    # If b is a top-level blossom, blossomparent[b] is -1.
    blossomparent = (2 * nvertex) * [-1]

    # If b is a non-trivial (sub-)blossom,
    # blossomchilds[b] is an ordered list of its sub-blossoms, starting with
    # the base and going round the blossom.
    blossomchilds = (2 * nvertex) * [None]

    # If b is a (sub-)blossom,
    # blossombase[b] is its base VERTEX (i.e. recursive sub-blossom).
    blossombase = list(range(nvertex)) + nvertex * [-1]

    # If b is a non-trivial (sub-)blossom,
    # blossomendps[b] is a list of endpoints on its connecting edges,
    # such that blossomendps[b][i] is the local endpoint of blossomchilds[b][i]
    # on the edge that connects it to blossomchilds[b][wrap(i+1)].
    blossomendps = (2 * nvertex) * [None]

    # If v is a free vertex (or an unreached vertex inside a T-blossom),
    # bestedge[v] is the edge to an S-vertex with least slack,
    # or -1 if there is no such edge.
    # If b is a (possibly trivial) top-level S-blossom,
    # bestedge[b] is the least-slack edge to a different S-blossom,
    # or -1 if there is no such edge.
    # This is used for efficient computation of delta2 and delta3.
    bestedge = (2 * nvertex) * [-1]

    # If b is a non-trivial top-level S-blossom,
    # blossombestedges[b] is a list of least-slack edges to neighbouring
    # S-blossoms, or None if no such list has been computed yet.
    # This is used for efficient computation of delta3.
    blossombestedges = (2 * nvertex) * [None]

    # List of currently unused blossom numbers.
    unusedblossoms = list(range(nvertex, 2 * nvertex))

    # If v is a vertex,
    # dualvar[v] = 2 * u(v) where u(v) is the v's variable in the dual
    # optimization problem (multiplication by two ensures integer values
    # throughout the algorithm if all edge weights are integers).
    # If b is a non-trivial blossom,
    # dualvar[b] = z(b) where z(b) is b's variable in the dual optimization
    # problem.
    dualvar = nvertex * [maxweight] + nvertex * [0]

    # If allowedge[k] is true, edge k has zero slack in the optimization
    # problem; if allowedge[k] is false, the edge's slack may or may not
    # be zero.
    allowedge = nedge * [False]

    # Queue of newly discovered S-vertices.
    queue = []

    # Return 2 * slack of edge k (does not work inside blossoms).
    def slack(k):
        (i, j, wt) = edges[k]
        return dualvar[i] + dualvar[j] - 2 * wt

    # Generate the leaf vertices of a blossom.
    def blossomLeaves(b):
        if b < nvertex:
            yield b
        else:
            for t in blossomchilds[b]:
                if t < nvertex:
                    yield t
                else:
                    for v in blossomLeaves(t):
                        yield v

    # Assign label t to the top-level blossom containing vertex w
    # and record the fact that w was reached through the edge with
    # remote endpoint p.
    def assignLabel(w, t, p):
        b = inblossom[w]
        assert label[w] == 0 and label[b] == 0
        label[w] = label[b] = t
        labelend[w] = labelend[b] = p
        bestedge[w] = bestedge[b] = -1
        if t == 1:
            # b became an S-vertex/blossom; add it(s vertices) to the queue.
            queue.extend(blossomLeaves(b))
        elif t == 2:
            # b became a T-vertex/blossom; assign label S to its mate.
            # (If b is a non-trivial blossom, its base is the only vertex
            # with an external mate.)
            base = blossombase[b]
            assert mate[base] >= 0
            assignLabel(endpoint[mate[base]], 1, mate[base] ^ 1)

    # Trace back from vertices v and w to discover either a new blossom
    # or an augmenting path. Return the base vertex of the new blossom or -1.
    def scanBlossom(v, w):
        # Trace back from v and w, placing breadcrumbs as we go.
        path = []
        base = -1
        while v != -1 or w != -1:
            # Look for a breadcrumb in v's blossom or put a new breadcrumb.
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            # Trace one step back.
            assert labelend[b] == mate[blossombase[b]]
            if labelend[b] == -1:
                # The base of blossom b is single; stop tracing this path.
                v = -1
            else:
                v = endpoint[labelend[b]]
                b = inblossom[v]
                assert label[b] == 2
                # b is a T-blossom; trace one more step back.
                assert labelend[b] >= 0
                v = endpoint[labelend[b]]
            # Swap v and w so that we alternate between both paths.
            if w != -1:
                v, w = w, v
        # Remove breadcrumbs.
        for b in path:
            label[b] = 1
        # Return base vertex, if we found one.
        return base

    # Construct a new blossom with given base, containing edge k which
    # connects a pair of S vertices. Label the new blossom as S; set its dual
    # variable to zero; relabel its T-vertices to S and add them to the queue.
    def addBlossom(base, k):
        (v, w, wt) = edges[k]
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        # Create blossom.
        b = unusedblossoms.pop()
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        # Make list of sub-blossoms and their interconnecting edge endpoints.
        blossomchilds[b] = path = []
        blossomendps[b] = endps = []
        # Trace back from v to base.
        while bv != bb:
            # Add bv to the new blossom.
            blossomparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            assert (label[bv] == 2 or
                    (label[bv] == 1 and labelend[bv] == mate[blossombase[bv]]))
            # Trace one step back.
            assert labelend[bv] >= 0
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        # Reverse lists, add endpoint that connects the pair of S vertices.
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        # Trace back from w to base.
        while bw != bb:
            # Add bw to the new blossom.
            blossomparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            assert (label[bw] == 2 or
                    (label[bw] == 1 and labelend[bw] == mate[blossombase[bw]]))
            # Trace one step back.
            assert labelend[bw] >= 0
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        # Set label to S.
        assert label[bb] == 1
        label[b] = 1
        labelend[b] = labelend[bb]
        # Set dual variable to zero.
        dualvar[b] = 0
        # Relabel vertices.
        for v in blossomLeaves(b):
            if label[inblossom[v]] == 2:
                # This T-vertex now turns into an S-vertex because it becomes
                # part of an S-blossom; add it to the queue.
                queue.append(v)
            inblossom[v] = b
        # Compute blossombestedges[b].
        bestedgeto = (2 * nvertex) * [-1]
        for bv in path:
            if blossombestedges[bv] is None:
                # This subblossom does not have a list of least-slack edges;
                # get the information from the vertices.
                nblists = [[p // 2 for p in neighbend[v]]
                           for v in blossomLeaves(bv)]
            else:
                # Walk this subblossom's least-slack edges.
                nblists = [blossombestedges[bv]]
            for nblist in nblists:
                for k in nblist:
                    (i, j, wt) = edges[k]
                    if inblossom[j] == b:
                        i, j = j, i
                    bj = inblossom[j]
                    if (bj != b and label[bj] == 1
                            and (bestedgeto[bj] == -1
                                 or slack(k) < slack(bestedgeto[bj]))):
                        bestedgeto[bj] = k
            # Forget about least-slack edges of the subblossom.
            blossombestedges[bv] = None
            bestedge[bv] = -1
        blossombestedges[b] = [k for k in bestedgeto if k != -1]
        # Select bestedge[b].
        bestedge[b] = -1
        for k in blossombestedges[b]:
            if bestedge[b] == -1 or slack(k) < slack(bestedge[b]):
                bestedge[b] = k

    # Expand the given top-level blossom.
    def expandBlossom(b, endstage):
        # Convert sub-blossoms into top-level blossoms.
        for s in blossomchilds[b]:
            blossomparent[s] = -1
            if s < nvertex:
                inblossom[s] = s
            elif endstage and dualvar[s] == 0:
                # Recursively expand this sub-blossom.
                expandBlossom(s, endstage)
            else:
                for v in blossomLeaves(s):
                    inblossom[v] = s
        # If we expand a T-blossom during a stage, its sub-blossoms must be
        # relabeled.
        if (not endstage) and label[b] == 2:
            # Start at the sub-blossom through which the expanding
            # blossom obtained its label, and relabel sub-blossoms untili
            # we reach the base.
            # Figure out through which sub-blossom the expanding blossom
            # obtained its label initially.
            assert labelend[b] >= 0
            entrychild = inblossom[endpoint[labelend[b] ^ 1]]
            # Decide in which direction we will go round the blossom.
            j = blossomchilds[b].index(entrychild)
            if j & 1:
                # Start index is odd; go forward and wrap.
                j -= len(blossomchilds[b])
                jstep = 1
                endptrick = 0
            else:
                # Start index is even; go backward.
                jstep = -1
                endptrick = 1
            # Move along the blossom until we get to the base.
            p = labelend[b]
            while j != 0:
                # Relabel the T-sub-blossom.
                label[endpoint[p ^ 1]] = 0
                label[endpoint[blossomendps[b][j - endptrick] ^ endptrick
                               ^ 1]] = 0
                assignLabel(endpoint[p ^ 1], 2, p)
                # Step to the next S-sub-blossom and note its forward endpoint.
                allowedge[blossomendps[b][j - endptrick] // 2] = True
                j += jstep
                p = blossomendps[b][j - endptrick] ^ endptrick
                # Step to the next T-sub-blossom.
                allowedge[p // 2] = True
                j += jstep
            # Relabel the base T-sub-blossom WITHOUT stepping through to
            # its mate (so don't call assignLabel).
            bv = blossomchilds[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            labelend[endpoint[p ^ 1]] = labelend[bv] = p
            bestedge[bv] = -1
            # Continue along the blossom until we get back to entrychild.
            j += jstep
            while blossomchilds[b][j] != entrychild:
                # Examine the vertices of the sub-blossom to see whether
                # it is reachable from a neighbouring S-vertex outside the
                # expanding blossom.
                bv = blossomchilds[b][j]
                if label[bv] == 1:
                    # This sub-blossom just got label S through one of its
                    # neighbours; leave it.
                    j += jstep
                    continue
                for v in blossomLeaves(bv):
                    if label[v] != 0:
                        break
                # If the sub-blossom contains a reachable vertex, assign
                # label T to the sub-blossom.
                if label[v] != 0:
                    assert label[v] == 2
                    assert inblossom[v] == bv
                    label[v] = 0
                    label[endpoint[mate[blossombase[bv]]]] = 0
                    assignLabel(v, 2, labelend[v])
                j += jstep
        # Recycle the blossom number.
        label[b] = labelend[b] = -1
        blossomchilds[b] = blossomendps[b] = None
        blossombase[b] = -1
        blossombestedges[b] = None
        bestedge[b] = -1
        unusedblossoms.append(b)

    # Swap matched/unmatched edges over an alternating path through blossom b
    # between vertex v and the base vertex.
    # Keep blossom bookkeeping consistent.
    def augmentBlossom(b, v):
        # Bubble up through the blossom tree from vertex v to an immediate
        # sub-blossom of b.
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        # Recursively deal with the first sub-blossom.
        if t >= nvertex:
            augmentBlossom(t, v)
        # Decide in which direction we will go round the blossom.
        i = j = blossomchilds[b].index(t)
        if i & 1:
            # Start index is odd; go forward and wrap.
            j -= len(blossomchilds[b])
            jstep = 1
            endptrick = 0
        else:
            # Start index is even; go backward.
            jstep = -1
            endptrick = 1
        # Move along the blossom until we get to the base.
        while j != 0:
            # Step to the next sub-blossom and augment it recursively.
            j += jstep
            t = blossomchilds[b][j]
            p = blossomendps[b][j - endptrick] ^ endptrick
            if t >= nvertex:
                augmentBlossom(t, endpoint[p])
            # Step to the next sub-blossom and augment it recursively.
            j += jstep
            t = blossomchilds[b][j]
            if t >= nvertex:
                augmentBlossom(t, endpoint[p ^ 1])
            # Match the edge connecting those sub-blossoms.
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
        # Rotate the list of sub-blossoms to put the new base at the front.
        blossomchilds[b] = blossomchilds[b][i:] + blossomchilds[b][:i]
        blossomendps[b] = blossomendps[b][i:] + blossomendps[b][:i]
        blossombase[b] = blossombase[blossomchilds[b][0]]
        assert blossombase[b] == v

    # Swap matched/unmatched edges over an alternating path between two
    # single vertices. The augmenting path runs through edge k, which
    # connects a pair of S vertices.
    def augmentMatching(k):
        (v, w, wt) = edges[k]
        for (s, p) in ((v, 2 * k + 1), (w, 2 * k)):
            # Match vertex s to remote endpoint p. Then trace back from s
            # until we find a single vertex, swapping matched and unmatched
            # edges as we go.
            while 1:
                bs = inblossom[s]
                assert label[bs] == 1
                assert labelend[bs] == mate[blossombase[bs]]
                # Augment through the S-blossom from s to base.
                if bs >= nvertex:
                    augmentBlossom(bs, s)
                # Update mate[s]
                mate[s] = p
                # Trace one step back.
                if labelend[bs] == -1:
                    # Reached single vertex; stop.
                    break
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                assert label[bt] == 2
                # Trace one step back.
                assert labelend[bt] >= 0
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                # Augment through the T-blossom from j to base.
                assert blossombase[bt] == t
                if bt >= nvertex:
                    augmentBlossom(bt, j)
                # Update mate[j]
                mate[j] = labelend[bt]
                # Keep the opposite endpoint;
                # it will be assigned to mate[s] in the next step.
                p = labelend[bt] ^ 1

    # Main loop: continue until no further improvement is possible.
    for t in range(nvertex):
        # Each iteration of this loop is a "stage".
        # A stage finds an augmenting path and uses that to improve
        # Remove labels from top-level blossoms/vertices.
        label[:] = (2 * nvertex) * [0]

        # Forget all about least-slack edges.
        bestedge[:] = (2 * nvertex) * [-1]
        blossombestedges[nvertex:] = nvertex * [None]

        # Loss of labeling means that we can not be sure that currently
        # allowable edges remain allowable throughout this stage.
        allowedge[:] = nedge * [False]

        # Make queue empty.
        queue[:] = []

        # Label single blossoms/vertices with S and put them in the queue.
        for v in range(nvertex):
            if mate[v] == -1 and label[inblossom[v]] == 0:
                assignLabel(v, 1, -1)

        # Loop until we succeed in augmenting the matching.
        augmented = 0
        while 1:

            # Each iteration of this loop is a "substage".
            # A substage tries to find an augmenting path;
            # if found, the path is used to improve the matching and
            # the stage ends. If there is no augmenting path, the
            # primal-dual method is used to pump some slack out of
            # the dual variables.

            # Continue labeling until all vertices which are reachable
            # through an alternating path have got a label.
            while queue and not augmented:

                # Take an S vertex from the queue.
                v = queue.pop()
                assert label[inblossom[v]] == 1

                # Scan its neighbours:
                for p in neighbend[v]:
                    k = p // 2
                    w = endpoint[p]
                    # w is a neighbour to v
                    if inblossom[v] == inblossom[w]:
                        # this edge is internal to a blossom; ignore it
                        continue
                    if not allowedge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            # edge k has zero slack => it is allowable
                            allowedge[k] = True
                    if allowedge[k]:
                        if label[inblossom[w]] == 0:
                            # (C1) w is a free vertex;
                            # label w with T and label its mate with S (R12).
                            assignLabel(w, 2, p ^ 1)
                        elif label[inblossom[w]] == 1:
                            # (C2) w is an S-vertex (not in the same blossom);
                            # follow back-links to discover either an
                            # augmenting path or a new blossom.
                            base = scanBlossom(v, w)
                            if base >= 0:
                                # Found a new blossom; add it to the blossom
                                # bookkeeping and turn it into an S-blossom.
                                addBlossom(base, k)
                            else:
                                # Found an augmenting path; augment the
                                # matching and end this stage.
                                augmentMatching(k)
                                augmented = 1
                                break
                        elif label[w] == 0:
                            # w is inside a T-blossom, but w itself has not
                            # yet been reached from outside the blossom;
                            # mark it as reached (we need this to relabel
                            # during T-blossom expansion).
                            assert label[inblossom[w]] == 2
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inblossom[w]] == 1:
                        # keep track of the least-slack non-allowable edge to
                        # a different S-blossom.
                        b = inblossom[v]
                        if bestedge[b] == -1 or kslack < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        # w is a free vertex (or an unreached vertex inside
                        # a T-blossom) but we can not reach it yet;
                        # keep track of the least-slack edge that reaches w.
                        if bestedge[w] == -1 or kslack < slack(bestedge[w]):
                            bestedge[w] = k

            if augmented:
                break

            # There is no augmenting path under these constraints;
            # compute delta and reduce slack in the optimization problem.
            # (Note that our vertex dual variables, edge slacks and delta's
            # are pre-multiplied by two.)
            deltatype = -1
            delta = deltaedge = deltablossom = None

            # Compute delta1: the minimum value of any vertex dual.
            if not maxcardinality:
                deltatype = 1
                delta = min(dualvar[:nvertex])

            # Compute delta2: the minimum slack on any edge between
            # an S-vertex and a free vertex.
            for v in range(nvertex):
                if label[inblossom[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]

            # Compute delta3: half the minimum slack on any edge between
            # a pair of S-blossoms.
            for b in range(2 * nvertex):
                if (blossomparent[b] == -1 and label[b] == 1
                        and bestedge[b] != -1):
                    kslack = slack(bestedge[b])
                    if isinstance(kslack, (int, )):
                        assert (kslack % 2) == 0
                        d = kslack // 2
                    else:
                        d = kslack / 2
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]

            # Compute delta4: minimum z variable of any T-blossom.
            for b in range(nvertex, 2 * nvertex):
                if (blossombase[b] >= 0 and blossomparent[b] == -1
                        and label[b] == 2
                        and (deltatype == -1 or dualvar[b] < delta)):
                    delta = dualvar[b]
                    deltatype = 4
                    deltablossom = b

            if deltatype == -1:
                # No further improvement possible; max-cardinality optimum
                # reached. Do a final delta update to make the optimum
                # verifyable.
                assert maxcardinality
                deltatype = 1
                delta = max(0, min(dualvar[:nvertex]))

            # Update dual variables according to delta.
            for v in range(nvertex):
                if label[inblossom[v]] == 1:
                    # S-vertex: 2*u = 2*u - 2*delta
                    dualvar[v] -= delta
                elif label[inblossom[v]] == 2:
                    # T-vertex: 2*u = 2*u + 2*delta
                    dualvar[v] += delta
            for b in range(nvertex, 2 * nvertex):
                if blossombase[b] >= 0 and blossomparent[b] == -1:
                    if label[b] == 1:
                        # top-level S-blossom: z = z + 2*delta
                        dualvar[b] += delta
                    elif label[b] == 2:
                        # top-level T-blossom: z = z - 2*delta
                        dualvar[b] -= delta

            # Take action at the point where minimum delta occurred.
            if deltatype == 1:
                # No further improvement possible; optimum reached.
                break
            elif deltatype == 2:
                # Use the least-slack edge to continue the search.
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                if label[inblossom[i]] == 0:
                    i, j = j, i
                assert label[inblossom[i]] == 1
                queue.append(i)
            elif deltatype == 3:
                # Use the least-slack edge to continue the search.
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                assert label[inblossom[i]] == 1
                queue.append(i)
            elif deltatype == 4:
                # Expand the least-z blossom.
                expandBlossom(deltablossom, False)

            # End of a this substage.

        # Stop when no more augmenting path can be found.
        if not augmented:
            break

        # End of a stage; expand all S-blossoms which have dualvar = 0.
        for b in range(nvertex, 2 * nvertex):
            if (blossomparent[b] == -1 and blossombase[b] >= 0
                    and label[b] == 1 and dualvar[b] == 0):
                expandBlossom(b, True)

    # Transform mate[] such that mate[v] is the vertex to which v is paired.
    for v in range(nvertex):
        if mate[v] >= 0:
            mate[v] = endpoint[mate[v]]
    for v in range(nvertex):
        assert mate[v] == -1 or mate[mate[v]] == v

    return mate


if __name__ == '__main__':
    database = Database(
        '/data/YuYoujiang/dataset/CrowdHuman/annotation_val.odgt',
        '/data/YuYoujiang/CrowdDet/mmdetection/tools/result_meg_scale.json',
        'box', None, 0)
    database.compare()
    eval_results = OrderedDict()
    AP, recall, _ = database.eval_AP()
    eval_results['mAP'] = float(f'{round(AP, 4)}')
    eval_results['recall'] = float(f'{round(recall, 4)}')
    mMR, _ = database.eval_MR()
    eval_results['mMR'] = float(f'{round(mMR, 4)}')
    _, JI = database.eval_JI()
    eval_results['JI'] = float(f'{round(JI, 4)}')
    print(eval_results)

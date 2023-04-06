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
from mmengine.fileio import dump, get_text, load
from mmengine.logging import MMLogger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.registry import METRICS

PERSON_CLASSES = ['background', 'person']


@METRICS.register_module()
class CrowdHumanMetric(BaseMetric):
    """CrowdHuman evaluation metric.

    Evaluate Average Precision (AP), Miss Rate (MR) and Jaccard Index (JI)
    for detection tasks.

    Args:
        ann_file (str): Path to the annotation file.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'AP', 'MR' and 'JI'. Defaults to 'AP'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
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
        eval_mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
        iou_thres (float): IoU threshold. Defaults to 0.5.
        compare_matching_method (str, optional): Matching method to compare
            the detection results with the ground_truth when compute 'AP'
            and 'MR'.Valid method include VOC and None(CALTECH). Default to
            None.
        mr_ref (str): Different parameter selection to calculate MR. Valid
            ref include CALTECH_-2 and CALTECH_-4. Defaults to CALTECH_-2.
        num_ji_process (int): The number of processes to evaluation JI.
            Defaults to 10.
    """
    default_prefix: Optional[str] = 'crowd_human'

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = ['AP', 'MR', 'JI'],
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 eval_mode: int = 0,
                 iou_thres: float = 0.5,
                 compare_matching_method: Optional[str] = None,
                 mr_ref: str = 'CALTECH_-2',
                 num_ji_process: int = 10) -> None:
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
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        assert eval_mode in [0, 1, 2], \
            "Unknown eval mode. mr_ref should be one of '0', '1', '2'."
        assert compare_matching_method is None or \
               compare_matching_method == 'VOC', \
               'The alternative compare_matching_method is VOC.' \
               'This parameter defaults to CALTECH(None)'
        assert mr_ref == 'CALTECH_-2' or mr_ref == 'CALTECH_-4', \
            "mr_ref should be one of 'CALTECH_-2', 'CALTECH_-4'."
        self.eval_mode = eval_mode
        self.iou_thres = iou_thres
        self.compare_matching_method = compare_matching_method
        self.mr_ref = mr_ref
        self.num_ji_process = num_ji_process

    @staticmethod
    def results2json(results: Sequence[dict], outfile_prefix: str) -> str:
        """Dump the detection results to a json file."""
        result_file_path = f'{outfile_prefix}.json'
        bbox_json_results = []
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
            bbox_json_results.append(dump_dict)
        dump(bbox_json_results, result_file_path)
        return result_file_path

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
            ann = dict()
            ann['ID'] = data_sample['img_id']
            ann['width'] = data_sample['ori_shape'][1]
            ann['height'] = data_sample['ori_shape'][0]
            pred_bboxes = data_sample['pred_instances']['bboxes'].cpu().numpy()
            pred_scores = data_sample['pred_instances']['scores'].cpu().numpy()

            pred_bbox_scores = np.hstack(
                [pred_bboxes, pred_scores.reshape((-1, 1))])

            self.results.append((ann, pred_bbox_scores))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            eval_results(Dict[str, float]): The computed metrics.
            The keys are the names of the metrics, and the values
            are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'result')
        else:
            outfile_prefix = self.outfile_prefix

        # convert predictions to coco format and dump to json file
        result_file = self.results2json(results, outfile_prefix)
        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        # load evaluation samples
        eval_samples = self.load_eval_samples(result_file)

        if 'AP' in self.metrics or 'MR' in self.metrics:
            score_list = self.compare(eval_samples)
            gt_num = sum([eval_samples[i].gt_num for i in eval_samples])
            ign_num = sum([eval_samples[i].ign_num for i in eval_samples])
            gt_num = gt_num - ign_num
            img_num = len(eval_samples)

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')
            if metric == 'AP':
                AP = self.eval_ap(score_list, gt_num, img_num)
                eval_results['mAP'] = float(f'{round(AP, 4)}')
            if metric == 'MR':
                MR = self.eval_mr(score_list, gt_num, img_num)
                eval_results['mMR'] = float(f'{round(MR, 4)}')
            if metric == 'JI':
                JI = self.eval_ji(eval_samples)
                eval_results['JI'] = float(f'{round(JI, 4)}')
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def load_eval_samples(self, result_file):
        """Load data from annotations file and detection results.

        Args:
            result_file (str): The file path of the saved detection results.

        Returns:
            Dict[Image]: The detection result packaged by Image
        """
        gt_str = get_text(
            self.ann_file, backend_args=self.backend_args).strip().split('\n')
        gt_records = [json.loads(line) for line in gt_str]

        pred_records = load(result_file, backend_args=self.backend_args)
        eval_samples = dict()
        for gt_record, pred_record in zip(gt_records, pred_records):
            assert gt_record['ID'] == pred_record['ID'], \
                'please set val_dataloader.sampler.shuffle=False and try again'
            eval_samples[pred_record['ID']] = Image(self.eval_mode)
            eval_samples[pred_record['ID']].load(gt_record, 'box', None,
                                                 PERSON_CLASSES, True)
            eval_samples[pred_record['ID']].load(pred_record, 'box', None,
                                                 PERSON_CLASSES, False)
            eval_samples[pred_record['ID']].clip_all_boader()
        return eval_samples

    def compare(self, samples):
        """Match the detection results with the ground_truth.

        Args:
            samples (dict[Image]): The detection result packaged by Image.

        Returns:
            score_list(list[tuple[ndarray, int, str]]): Matching result.
            a list of tuples (dtbox, label, imgID) in the descending
            sort of dtbox.score.
        """
        score_list = list()
        for id in samples:
            if self.compare_matching_method == 'VOC':
                result = samples[id].compare_voc(self.iou_thres)
            else:
                result = samples[id].compare_caltech(self.iou_thres)
            score_list.extend(result)
        # In the descending sort of dtbox score.
        score_list.sort(key=lambda x: x[0][-1], reverse=True)
        return score_list

    @staticmethod
    def eval_ap(score_list, gt_num, img_num):
        """Evaluate by average precision.

        Args:
            score_list(list[tuple[ndarray, int, str]]): Matching result.
                a list of tuples (dtbox, label, imgID) in the descending
                sort of dtbox.score.
            gt_num(int): The number of gt boxes in the entire dataset.
            img_num(int)： The number of images in the entire dataset.

        Returns:
            ap(float): result of average precision.
        """

        # calculate general ap score
        def _calculate_map(_recall, _precision):
            assert len(_recall) == len(_precision)
            area = 0
            for k in range(1, len(_recall)):
                delta_h = (_precision[k - 1] + _precision[k]) / 2
                delta_w = _recall[k] - _recall[k - 1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list()

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in enumerate(score_list):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = gt_num - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / img_num)

        ap = _calculate_map(rpX, rpY)
        return ap

    def eval_mr(self, score_list, gt_num, img_num):
        """Evaluate by Caltech-style log-average miss rate.

        Args:
            score_list(list[tuple[ndarray, int, str]]): Matching result.
                a list of tuples (dtbox, label, imgID) in the descending
                sort of dtbox.score.
            gt_num(int): The number of gt boxes in the entire dataset.
            img_num(int): The number of image in the entire dataset.

        Returns:
            mr(float): result of miss rate.
        """

        # find greater_than
        def _find_gt(lst, target):
            for idx, _item in enumerate(lst):
                if _item >= target:
                    return idx
            return len(lst) - 1

        if self.mr_ref == 'CALTECH_-2':
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

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list()
        for i, item in enumerate(score_list):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = gt_num - tp
            recall = tp / (tp + fn)
            missrate = 1.0 - recall
            fppi = fp / img_num
            fppiX.append(fppi)
            fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        mr = np.exp(np.log(score).mean())
        return mr

    def eval_ji(self, samples):
        """Evaluate by JI using multi_process.

        Args:
            samples(Dict[str, Image]): The detection result packaged by Image.

        Returns:
            ji(float): result of jaccard index.
        """
        import math
        res_line = []
        res_ji = []
        for i in range(10):
            score_thr = 1e-1 * i
            total = len(samples)
            stride = math.ceil(total / self.num_ji_process)
            result_queue = Queue(10000)
            results, procs = [], []
            records = list(samples.items())
            for i in range(self.num_ji_process):
                start = i * stride
                end = np.min([start + stride, total])
                sample_data = dict(records[start:end])
                p = Process(
                    target=self.compute_ji_with_ignore,
                    args=(result_queue, sample_data, score_thr))
                p.start()
                procs.append(p)
            for i in range(total):
                t = result_queue.get()
                results.append(t)
            for p in procs:
                p.join()
            line, mean_ratio = self.gather(results)
            line = 'score_thr:{:.1f}, {}'.format(score_thr, line)
            res_line.append(line)
            res_ji.append(mean_ratio)
        return max(res_ji)

    def compute_ji_with_ignore(self, result_queue, dt_result, score_thr):
        """Compute JI with ignore.

        Args:
            result_queue(Queue): The Queue for save compute result when
                multi_process.
            dt_result(dict[Image]): Detection result packaged by Image.
            score_thr(float): The threshold of detection score.
        Returns:
            dict: compute result.
        """
        for ID, record in dt_result.items():
            gt_boxes = record.gt_boxes
            dt_boxes = record.dt_boxes
            keep = dt_boxes[:, -1] > score_thr
            dt_boxes = dt_boxes[keep][:, :-1]

            gt_tag = np.array(gt_boxes[:, -1] != -1)
            matches = self.compute_ji_matching(dt_boxes, gt_boxes[gt_tag, :4])
            # get the unmatched_indices
            matched_indices = np.array([j for (j, _) in matches])
            unmatched_indices = list(
                set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
            num_ignore_dt = self.get_ignores(dt_boxes[unmatched_indices],
                                             gt_boxes[~gt_tag, :4])
            matched_indices = np.array([j for (_, j) in matches])
            unmatched_indices = list(
                set(np.arange(gt_boxes[gt_tag].shape[0])) -
                set(matched_indices))
            num_ignore_gt = self.get_ignores(
                gt_boxes[gt_tag][unmatched_indices], gt_boxes[~gt_tag, :4])
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

    @staticmethod
    def gather(results):
        """Integrate test results."""
        assert len(results)
        img_num = 0
        for result in results:
            if result['n'] != 0 or result['m'] != 0:
                img_num += 1
        mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
        valids = np.sum([rb['k'] for rb in results])
        total = np.sum([rb['n'] for rb in results])
        gtn = np.sum([rb['m'] for rb in results])
        line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}'\
            .format(mean_ratio, valids, total, gtn)
        return line, mean_ratio

    def compute_ji_matching(self, dt_boxes, gt_boxes):
        """Match the annotation box for each detection box.

        Args:
            dt_boxes(ndarray): Detection boxes.
            gt_boxes(ndarray): Ground_truth boxes.

        Returns:
            matches_(list[tuple[int, int]]): Match result.
        """
        assert dt_boxes.shape[-1] > 3 and gt_boxes.shape[-1] > 3
        if dt_boxes.shape[0] < 1 or gt_boxes.shape[0] < 1:
            return list()

        ious = bbox_overlaps(dt_boxes, gt_boxes, mode='iou')
        input_ = copy.deepcopy(ious)
        input_[input_ < self.iou_thres] = 0
        match_scipy = maximum_bipartite_matching(
            csr_matrix(input_), perm_type='column')
        matches_ = []
        for i in range(len(match_scipy)):
            if match_scipy[i] != -1:
                matches_.append((i, int(match_scipy[i])))
        return matches_

    def get_ignores(self, dt_boxes, gt_boxes):
        """Get the number of ignore bboxes."""
        if gt_boxes.size:
            ioas = bbox_overlaps(dt_boxes, gt_boxes, mode='iof')
            ioas = np.max(ioas, axis=1)
            rows = np.where(ioas > self.iou_thres)[0]
            return len(rows)
        else:
            return 0


class Image(object):
    """Data structure for evaluation of CrowdHuman.

    Note:
        This implementation is modified from https://github.com/Purkialo/
        CrowdDet/blob/master/lib/evaluate/APMRToolkits/image.py

    Args:
        mode (int): Select the mode of evaluate. Valid mode include
            0(just body box), 1(just head box) and 2(both of them).
            Defaults to 0.
    """

    def __init__(self, mode):
        self.ID = None
        self.width = None
        self.height = None
        self.dt_boxes = None
        self.gt_boxes = None
        self.eval_mode = mode

        self.ign_num = None
        self.gt_num = None
        self.dt_num = None

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
            body_key (str, None): key of detection body box.
                Valid when loading detection results and self.eval_mode!=1.
            head_key (str, None): key of detection head box.
                Valid when loading detection results and self.eval_mode!=0.
            class_names (list[str]):class names of data set.
                Defaults to ['background', 'person'].
            gt_flag (bool): Indicate whether record is ground truth
                or predicting the outcome.
        """
        if 'ID' in record and self.ID is None:
            self.ID = record['ID']
        if 'width' in record and self.width is None:
            self.width = record['width']
        if 'height' in record and self.height is None:
            self.height = record['height']
        if gt_flag:
            self.gt_num = len(record['gtboxes'])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes',
                                                      class_names)
            if self.eval_mode == 0:
                self.gt_boxes = body_bbox
                self.ign_num = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gt_boxes = head_bbox
                self.ign_num = (head_bbox[:, -1] == -1).sum()
            else:
                gt_tag = np.array([
                    body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                    for i in range(len(body_bbox))
                ])
                self.ign_num = (gt_tag == 0).sum()
                self.gt_boxes = np.hstack(
                    (body_bbox[:, :-1], head_bbox[:, :-1],
                     gt_tag.reshape(-1, 1)))

        if not gt_flag:
            self.dt_num = len(record['dtboxes'])
            if self.eval_mode == 0:
                self.dt_boxes = self.load_det_boxes(record, 'dtboxes',
                                                    body_key, 'score')
            elif self.eval_mode == 1:
                self.dt_boxes = self.load_det_boxes(record, 'dtboxes',
                                                    head_key, 'score')
            else:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key,
                                                   'score')
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key,
                                                   'score')
                self.dt_boxes = np.hstack((body_dtboxes, head_dtboxes))

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
        """load detection boxes."""
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

    def clip_all_boader(self):
        """Make sure boxes are within the image range."""

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dt_boxes.shape[-1] >= 4
        assert self.gt_boxes.shape[-1] >= 4
        assert self.width is not None and self.height is not None
        if self.eval_mode == 2:
            self.dt_boxes[:, :4] = _clip_boundary(self.dt_boxes[:, :4],
                                                  self.height, self.width)
            self.gt_boxes[:, :4] = _clip_boundary(self.gt_boxes[:, :4],
                                                  self.height, self.width)
            self.dt_boxes[:, 4:8] = _clip_boundary(self.dt_boxes[:, 4:8],
                                                   self.height, self.width)
            self.gt_boxes[:, 4:8] = _clip_boundary(self.gt_boxes[:, 4:8],
                                                   self.height, self.width)
        else:
            self.dt_boxes = _clip_boundary(self.dt_boxes, self.height,
                                           self.width)
            self.gt_boxes = _clip_boundary(self.gt_boxes, self.height,
                                           self.width)

    def compare_voc(self, thres):
        """Match the detection results with the ground_truth by VOC.

        Args:
            thres (float): IOU threshold.

        Returns:
            score_list(list[tuple[ndarray, int, str]]): Matching result.
            a list of tuples (dtbox, label, imgID) in the descending
            sort of dtbox.score.
        """
        if self.dt_boxes is None:
            return list()
        dtboxes = self.dt_boxes
        gtboxes = self.gt_boxes if self.gt_boxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        score_list = list()
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
                    score_list.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                score_list.append((dt, self.ID))
        return score_list

    def compare_caltech(self, thres):
        """Match the detection results with the ground_truth by Caltech
        matching strategy.

        Args:
            thres (float): IOU threshold.

        Returns:
            score_list(list[tuple[ndarray, int, str]]): Matching result.
            a list of tuples (dtbox, label, imgID) in the descending
            sort of dtbox.score.
        """
        if self.dt_boxes is None or self.gt_boxes is None:
            return list()

        dtboxes = self.dt_boxes if self.dt_boxes is not None else list()
        gtboxes = self.gt_boxes if self.gt_boxes is not None else list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = bbox_overlaps(dtboxes, gtboxes, mode='iou')
            overlap_ioa = bbox_overlaps(dtboxes, gtboxes, mode='iof')
        else:
            return list()

        score_list = list()
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
                    score_list.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                score_list.append((dt, 0, self.ID))
        return score_list

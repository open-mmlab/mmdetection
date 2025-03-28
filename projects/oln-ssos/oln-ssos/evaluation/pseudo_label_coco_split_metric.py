import itertools
import tempfile
from collections import OrderedDict
from typing import Optional, Sequence, Dict

import numpy as np
import torch
from mmengine import dump, MMLogger

from mmengine.fileio import load
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOevalMP
from mmdet.evaluation import CocoSplitMetric
from mmdet.evaluation.metrics.cocoeval_wrappers import COCOEvalXclassWrapper
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results

import os.path as osp


@METRICS.register_module()
class PseudoLabelCocoSplitMetric(CocoSplitMetric):

    def __init__(self, mode='id', optimal_score_threshold=0.0, anomaly_score_threshold=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.optimal_score_threshold = optimal_score_threshold
        self.anomaly_score_threshold = anomaly_score_threshold

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ood_scores = result['ood_scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['ood_score'] = float(ood_scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['ood_score'] = float(ood_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

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
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['ood_scores'] = pred['ood_scores'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
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

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                if self.mode == 'ood':
                    predictions = [p for p in predictions if p['score'] > self.optimal_score_threshold
                               and p['ood_score'] < self.anomaly_score_threshold]
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

                # Class manipulation.
            for idx, ann in enumerate(self._coco_api.dataset['annotations']):
                if ann['category_id'] in self.eval_cat_ids:
                    self._coco_api.dataset['annotations'][idx]['ignored_split'] = 0
                else:
                    self._coco_api.dataset['annotations'][idx]['ignored_split'] = 1

            if self.use_mp_eval:
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOEvalXclassWrapper(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.params.useCats = 0

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                if self.mode == 'id':
                    # Compute optimal micro F1 score threshold. We compute the f1 score for
                    # every class and score threshold. We then compute the score threshold that
                    # maximizes the F-1 score of every class. The final score threshold is the average
                    # over all classes.
                    # if optimal_score_threshold is None:
                    precisions = coco_eval.eval['precision'].mean(0)[:, :, 0, 2]
                    recalls = np.expand_dims(coco_eval.params.recThrs, 1)
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
                    optimal_f1_score = f1_scores.argmax(0)
                    scores = coco_eval.eval['scores'].mean(0)[:, :, 0, 2]
                    optimal_score_threshold = [scores[optimal_f1_score_i, i]
                                               for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
                    optimal_score_threshold = np.array(optimal_score_threshold)
                    optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
                    optimal_score_threshold = optimal_score_threshold.mean()
                    logger.info(f"Optimal score threshold: {optimal_score_threshold.item()}")

                    dt_ids_with_match = [int(dt_id) for ev_im in coco_eval.evalImgs for dt_id in ev_im['gtMatches'][0]
                                         if
                                         dt_id > 0]
                    dt_ids_with_match = list(set(dt_ids_with_match))
                    valid_detections = coco_eval.cocoDt.loadAnns(
                        dt_ids_with_match)  # list(results_api.cocoDt.anns.values())#
                    ood_scores = [o['ood_score'] for o in valid_detections]
                    ood_scores.sort()
                    anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
                    logger.info(f"Non OS Detected Anomaly Score Threshold: {anomaly_score_threshold}")
                    # data_to_print += f',{anomaly_score_threshold:.5f}'
                    valid_detections = coco_eval.cocoDt.anns.values()  # list(results_api.cocoDt.anns.values())#
                    ood_scores = [o['ood_score'] for o in valid_detections]
                    ood_scores.sort()
                    anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
                    logger.info(f"Non OS All Anomaly Score Threshold: {anomaly_score_threshold}")
                    # data_to_print += f',{anomaly_score_threshold:.5f}'
                    dt_ids_with_match = [int(dt_id) for ev_im in coco_eval.evalImgs for dt_id in ev_im['gtMatches'][0]
                                         if
                                         dt_id > 0]
                    dt_ids_with_match = list(set(dt_ids_with_match))
                    valid_detections = coco_eval.cocoDt.loadAnns(
                        dt_ids_with_match)  # list(results_api.cocoDt.anns.values())#
                    optimal_detections = [v for v in valid_detections if v['score'] > optimal_score_threshold]
                    ood_scores = [o['ood_score'] for o in optimal_detections]
                    ood_scores.sort()
                    anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
                    logger.info(f"Detected Anomaly Score Threshold: {anomaly_score_threshold}")
                    # data_to_print += f',{anomaly_score_threshold:.5f}'
                    valid_detections = coco_eval.cocoDt.anns.values()  # list(results_api.cocoDt.anns.values())#
                    optimal_detections = [v for v in valid_detections if v['score'] > optimal_score_threshold]
                    ood_scores = [o['ood_score'] for o in optimal_detections]
                    ood_scores.sort()
                    anomaly_score_threshold = ood_scores[int(len(ood_scores) * 0.05)]
                    logger.info(f"All Anomaly Score Threshold: {anomaly_score_threshold}")
                    # data_to_print += f',{anomaly_score_threshold:.5f}'

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

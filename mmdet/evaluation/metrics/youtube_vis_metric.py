# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
import zipfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.dist import (all_gather_object, barrier, broadcast_object_list,
                           is_main_process)
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import YTVIS, YTVISeval
from .base_video_metric import BaseVideoMetric, collect_tracking_results


@METRICS.register_module()
class YouTubeVISMetric(BaseVideoMetric):
    """mAP evaluation metrics for the VIS task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `youtube_vis_ap`.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonyms metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
    """

    default_prefix: Optional[str] = 'youtube_vis'

    def __init__(self,
                 metric: Union[str, List[str]] = 'youtube_vis_ap',
                 metric_items: Optional[Sequence[str]] = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # vis evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.format_only = format_only
        allowed_metrics = ['youtube_vis_ap']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    f"metric should be 'youtube_vis_ap', but got {metric}.")

        self.metric_items = metric_items
        self.outfile_prefix = outfile_prefix
        self.per_video_res = []
        self.categories = []
        self._vis_meta_info = defaultdict(list)  # record video and image infos

    def process_video(self, data_samples):

        video_length = len(data_samples)
        for frame_id in range(video_length):
            result = dict()
            img_data_sample = data_samples[frame_id].to_dict()
            pred = img_data_sample['pred_track_instances']
            video_id = img_data_sample['video_id']

            result['img_id'] = img_data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['instances_id'] = pred['instances_id'].cpu().numpy()
            # encode mask to RLE
            assert 'masks' in pred, \
                'masks must exist in YouTube-VIS metric'
            result['masks'] = encode_mask_results(
                pred['masks'].detach().cpu().numpy())

            # parse gt
            gt = dict()
            gt['width'] = img_data_sample['ori_shape'][1]
            gt['height'] = img_data_sample['ori_shape'][0]
            gt['img_id'] = img_data_sample['img_id']
            gt['frame_id'] = frame_id
            gt['video_id'] = video_id
            gt['video_length'] = video_length

            if 'instances' in img_data_sample:
                gt['anns'] = img_data_sample['instances']
            else:
                gt['anns'] = dict()
            self.per_video_res.append((result, gt))

        preds, gts = zip(*self.per_video_res)
        # format the results
        # we must format gts first to update self._vis_meta_info
        gt_results = self._format_one_video_gts(gts)
        pred_results = self._format_one_video_preds(preds)
        self.per_video_res.clear()
        # add converted result to the results list
        self.results.append((pred_results, gt_results))

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # split gt and prediction list
        tmp_pred_results, tmp_gt_results = zip(*results)
        gt_results = self.format_gts(tmp_gt_results)
        pred_results = self.format_preds(tmp_pred_results)

        if self.format_only:
            self.save_pred_results(pred_results)
            return dict()

        ytvis = YTVIS(gt_results)

        ytvis_dets = ytvis.loadRes(pred_results)
        vid_ids = ytvis.getVidIds()

        iou_type = metric = 'segm'
        eval_results = OrderedDict()
        ytvisEval = YTVISeval(ytvis, ytvis_dets, iou_type)
        ytvisEval.params.vidIds = vid_ids
        ytvisEval.evaluate()
        ytvisEval.accumulate()
        ytvisEval.summarize()

        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@1': 6,
            'AR@10': 7,
            'AR@100': 8,
            'AR_s@100': 9,
            'AR_m@100': 10,
            'AR_l@100': 11
        }
        metric_items = self.metric_items
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item "{metric_item}" is not supported')

        if metric_items is None:
            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{ytvisEval.stats[coco_metric_names[metric_item]]:.3f}')
            eval_results[key] = val

        return eval_results

    def format_gts(self, gts: Tuple[List]) -> dict:
        """Gather all ground-truth from self.results."""
        self.categories = [
            dict(id=id + 1, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        gt_results = dict(
            categories=self.categories,
            videos=self._vis_meta_info['videos'],
            annotations=[])
        for gt_result in gts:
            gt_results['annotations'].extend(gt_result)
        return gt_results

    def format_preds(self, preds: Tuple[List]) -> List:
        """Gather all predictions from self.results."""
        pred_results = []
        for pred_result in preds:
            pred_results.extend(pred_result)
        return pred_results

    def _format_one_video_preds(self, pred_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            pred_dicts (Tuple[dict]): Prediction of the dataset.

        Returns:
            List: The formatted predictions.
        """
        # Collate preds scatters (tuple of dict to dict of list)
        preds = defaultdict(list)
        for pred in pred_dicts:
            for key in pred.keys():
                preds[key].append(pred[key])

        img_infos = self._vis_meta_info['images']
        vid_infos = self._vis_meta_info['videos']
        inds = [i for i, _ in enumerate(img_infos) if _['frame_id'] == 0]
        inds.append(len(img_infos))
        json_results = []
        video_id = vid_infos[-1]['id']
        # collect data for each instances in a video.
        collect_data = dict()
        for frame_id, (masks, scores, labels, ids) in enumerate(
                zip(preds['masks'], preds['scores'], preds['labels'],
                    preds['instances_id'])):

            assert len(masks) == len(labels)
            for j, id in enumerate(ids):
                if id not in collect_data:
                    collect_data[id] = dict(
                        category_ids=[], scores=[], segmentations=dict())
                collect_data[id]['category_ids'].append(labels[j])
                collect_data[id]['scores'].append(scores[j])
                if isinstance(masks[j]['counts'], bytes):
                    masks[j]['counts'] = masks[j]['counts'].decode()
                collect_data[id]['segmentations'][frame_id] = masks[j]

        # transform the collected data into official format
        for id, id_data in collect_data.items():
            output = dict()
            output['video_id'] = video_id
            output['score'] = np.array(id_data['scores']).mean().item()
            # majority voting for sequence category
            output['category_id'] = np.bincount(
                np.array(id_data['category_ids'])).argmax().item() + 1
            output['segmentations'] = []
            for frame_id in range(inds[-1] - inds[-2]):
                if frame_id in id_data['segmentations']:
                    output['segmentations'].append(
                        id_data['segmentations'][frame_id])
                else:
                    output['segmentations'].append(None)
            json_results.append(output)

        return json_results

    def _format_one_video_gts(self, gt_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.

        This operation is to make it easier to use the official eval API.

        Args:
            gt_dicts (Tuple[dict]): Ground truth of the dataset.

        Returns:
            list: The formatted gts.
        """
        video_infos = []
        image_infos = []
        instance_infos = defaultdict(list)
        len_videos = dict()  # mapping from instance_id to video_length
        vis_anns = []

        # get video infos
        for gt_dict in gt_dicts:
            frame_id = gt_dict['frame_id']
            video_id = gt_dict['video_id']
            img_id = gt_dict['img_id']
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                frame_id=frame_id,
                file_name='')
            image_infos.append(image_info)
            if frame_id == 0:
                video_info = dict(
                    id=video_id,
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                    file_name='')
                video_infos.append(video_info)

            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                instance_id = ann['instance_id']
                # update video length
                len_videos[instance_id] = gt_dict['video_length']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    video_id=video_id,
                    frame_id=frame_id,
                    bbox=coco_bbox,
                    instance_id=instance_id,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label) + 1,
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask

                instance_infos[instance_id].append(annotation)

        # update vis meta info
        self._vis_meta_info['images'].extend(image_infos)
        self._vis_meta_info['videos'].extend(video_infos)

        for instance_id, ann_infos in instance_infos.items():
            cur_video_len = len_videos[instance_id]
            segm = [None] * cur_video_len
            bbox = [None] * cur_video_len
            area = [None] * cur_video_len
            # In the official format, no instances are represented by
            # 'None', however, only images with instances are recorded
            # in the current annotations, so we need to use 'None' to
            # initialize these lists.
            for ann_info in ann_infos:
                frame_id = ann_info['frame_id']
                segm[frame_id] = ann_info['segmentation']
                bbox[frame_id] = ann_info['bbox']
                area[frame_id] = ann_info['area']
            instance = dict(
                category_id=ann_infos[0]['category_id'],
                segmentations=segm,
                bboxes=bbox,
                video_id=ann_infos[0]['video_id'],
                areas=area,
                id=instance_id,
                iscrowd=ann_infos[0]['iscrowd'])
            vis_anns.append(instance)
        return vis_anns

    def save_pred_results(self, pred_results: List) -> None:
        """Save the results to a zip file (standard format for YouTube-VIS
        Challenge).

        Args:
            pred_results (list): Testing results of the
                dataset.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        mmengine.dump(pred_results, f'{outfile_prefix}.json')
        # zip the json file in order to submit to the test server.
        zip_file_name = f'{outfile_prefix}.submission_file.zip'
        zf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        logger.info(f"zip the 'results.json' into '{zip_file_name}', "
                    'please submmit the zip file to the test server')
        zf.write(f'{outfile_prefix}.json', 'results.json')
        zf.close()

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()

        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        results = collect_tracking_results(self.results, self.collect_device)

        # gather seq_info
        gathered_seq_info = all_gather_object(self._vis_meta_info['videos'])
        all_seq_info = []
        for _seq_info in gathered_seq_info:
            all_seq_info.extend(_seq_info)
        # update self._vis_meta_info
        self._vis_meta_info = dict(videos=all_seq_info)

        if is_main_process():
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        # reset the vis_meta_info
        self._vis_meta_info.clear()
        return metrics[0]

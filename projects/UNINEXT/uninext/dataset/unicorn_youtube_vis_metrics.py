# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_util

from mmdet.evaluation.metrics import YouTubeVISMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class YouTubeVISUNIMetric(YouTubeVISMetric):
    """mAP evaluation metrics for the VIS task.

    Args:
        is_multi_cls (bool): Default: True
        apply_cls_thres (float):  Default: 0.05
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
                 is_multi_cls: bool = True,
                 apply_cls_thres: float = 0.05,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

    def process_video(self, data_samples):

        video_len = len(data_samples)
        for frame_id in range(video_len):
            img_data_sample = data_samples[frame_id].to_dict()
            result = dict()
            pred = img_data_sample['pred_track_instances']
            video_id = img_data_sample['video_id']
            result['scores'] = pred['scores'].cpu().numpy()
            result['instances_id'] = pred['instances_id'].cpu().numpy()
            # encode mask to RLE
            assert 'masks' in pred, \
                'masks must exist in YouTube-VIS metric'
            result['masks'] = pred['masks'].detach().cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = img_data_sample['ori_shape'][1]
            gt['height'] = img_data_sample['ori_shape'][0]
            gt['frame_id'] = frame_id
            gt['video_id'] = video_id
            gt['video_length'] = video_len

            # When the ground truth exists, get annotation from `instances`.
            # In general, it contains `bbox`, `bbox_label`, `mask` and
            # `instance_id`.
            if 'instances' in img_data_sample:
                gt['anns'] = img_data_sample['instances']
            else:
                gt['anns'] = dict()

            self.per_video_res.append((result, gt))
        preds, gts = zip(*self.per_video_res)
        # format the results
        # we must format gts first to update self._vis_meta_info
        gt_results = self._format_one_video_gts(gts)
        pred_results = self._format_one_video_preds(preds, video_id, video_len)
        self.per_video_res.clear()
        # add converted result to the results list
        self.results.append((pred_results, gt_results))

    def _format_one_video_preds(self, pred_dicts: Tuple[dict], video_id,
                                video_len) -> List:
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

        json_results = []
        collect_data = dict()
        for frame_id, (masks, scores, ids) in enumerate(
                zip(preds['masks'], preds['scores'], preds['instances_id'])):

            assert len(masks) == len(scores)
            for j, id in enumerate(ids):
                if id not in collect_data:
                    collect_data[id] = dict(
                        valid=0,
                        start_frame=frame_id,
                        frame=[],
                        scores=[],
                        segmentations=dict())
                collect_data[id]['scores'].append(scores[j])
                collect_data[id]['valid'] = collect_data[id]['valid'] + 1
                collect_data[id]['frame'].append(frame_id)
                collect_data[id]['segmentations'][frame_id] = masks[j]

            if frame_id > 8:
                del_list = []
                for id in collect_data.keys():
                    if frame_id > collect_data[id]['start_frame'] and \
                        (frame_id - 1) not in collect_data[id]['frame'] and  \
                        (frame_id - 2) not in collect_data[id]['frame'] and  \
                            collect_data[id]['valid'] < 3:
                        del_list.append(id)
                for del_k in del_list:
                    collect_data.pop(del_k)

        logits_list = []
        masks_list = []
        # transform the collected data into official format
        for id, id_data in collect_data.items():
            masks_list_i = []
            logits_list.append(np.stack(id_data['scores']).mean(0))
            for frame_id in range(video_len):
                if frame_id in id_data['segmentations']:
                    masks_list_i.append(id_data['segmentations'][frame_id])
                else:
                    masks_list_i.append(None)
            masks_list.append(masks_list_i)

        if len(logits_list) > 0:
            pred_cls = np.stack(logits_list)
        else:
            pred_cls = []

        if len(pred_cls) > 0:
            if self.is_multi_cls:
                # is_above_thres is a tuple of 1-D tensors,
                # one for each dimension in input,
                is_above_thres = np.where(pred_cls > self.apply_cls_thres)
                scores = pred_cls[is_above_thres]  # (num_obj, )
                labels = is_above_thres[1]  # (num_obj, )
                masks_list_mc = []  # masks_list multi_cls
                for idx in is_above_thres[0]:
                    masks_list_mc.append(masks_list[idx])
                out_masks = masks_list_mc
            else:
                scores, labels = pred_cls.max(-1)
                out_masks = masks_list
            out_scores = scores.tolist()
            out_labels = labels.tolist()
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        for (s, l, m) in zip(out_scores, out_labels, out_masks):
            output = dict()
            output['video_id'] = video_id
            output['score'] = s
            output['category_id'] = l + 1
            segms = []
            for _mask in m:
                if _mask is None:
                    segms.append(None)
                else:
                    rle = mask_util.encode(
                        np.array(_mask[:, :, None], order='F',
                                 dtype='uint8'))[0]
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = rle['counts'].decode('utf-8')
                    segms.append(rle)
            output['segmentations'] = segms
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
            # img_id = gt_dict['img_id']
            image_info = dict(
                # id=img_id,
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

import logging
import os.path as osp
from typing import List, Tuple

from mmengine.fileio import get_local_path
from pycocotools.ytvos import YTVOS

from mmdet.datasets import YouTubeVISDataset
from mmdet.registry import DATASETS

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class UNIYtvisDataSet(YouTubeVISDataset):

    def __init__(self,
                 *args,
                 extra_annotation_keys: dict = None,
                 has_mask: bool = True,
                 has_expression: bool = False,
                 sot_task: bool = False,
                 dataset_name: str = 'ytvis_2019',
                 **kwargs) -> None:

        self.extra_annotations_keys = extra_annotation_keys
        self.has_mask = has_mask
        self.has_expression = has_expression
        self.sot_task = sot_task
        self.dataset_name = dataset_name
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        with get_local_path(self.ann_file) as local_path:
            self.ytvis_api = YTVOS(local_path)

        self.cat_ids = sorted(
            self.ytvis_api.getCatIds(catNms=self.metainfo['classes']))
        self.cats = self.ytvis_api.loadCats(self.cat_ids)
        # id_map (1 - 90) - (0 - 80)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        vid_ids = sorted(self.ytvis_api.vids.keys())
        vids = self.ytvis_api.loadVids(vid_ids)
        anns = [self.ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]

        vids_anns = list(zip(vids, anns))
        logger.info('Loaded {} videos in YTVIS format from {}'.format(
            len(vids_anns), self.ann_file))

        videos = {}

        num_instances_without_valid_segmentation = 0

        for (vid_dict, anno_dict_list) in vids_anns:
            video_id = vid_dict['id']
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'images': [],
                    'video_length': vid_dict['length']
                }
            for frame_idx in range(vid_dict['length']):
                frame_objs = []
                for anno in anno_dict_list:
                    assert anno['video_id'] == video_id
                    obj = {
                        'category_id': anno['category_id'],
                        'ignore_flag': 1 if anno('iscrowd', False) else 0,
                        'instance_id': anno['id']
                    }
                    _bboxes = anno.get('bboxes', None)
                    _segm = anno.get('segmentations', None)
                    if self.has_mask:
                        if not (_bboxes and _segm and _bboxes[frame_idx]
                                and _segm[frame_idx]):
                            continue
                    else:
                        if not (_bboxes and _bboxes[frame_idx]):
                            continue
                    if 'ori_id' in anno:
                        # for VOS inference
                        obj['ori_id'] = anno['ori_id']
                    bbox = _bboxes[frame_idx]
                    x1, y1, w, h = bbox
                    obj['bbox'] = [x1, y1, x1 + w, y1 + h]

                    if self.has_mask:
                        segm = _segm[frame_idx]
                        if not isinstance(segm, dict):
                            # filter out invalid polygons (< 3 points)
                            segm = [
                                poly for poly in segm
                                if len(poly) % 2 == 0 and len(poly) >= 6
                            ]
                            if len(segm) == 0:
                                num_instances_without_valid_segmentation += 1
                                continue  # ignore this instance
                        obj['mask'] = segm

                    if self.cat2label:
                        obj['bbox_label'] = self.cat2label[obj['category_id']]
                        obj.pop('category_id')
                    frame_objs.append(obj)
                parsed_data_info = self.parse_data_info(
                    dict(
                        raw_video_info=vid_dict,
                        raw_ann_info=frame_objs,
                        frame_idx=frame_idx))
                videos[video_id]['images'].append(parsed_data_info)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                'Filtered out {} instances without valid segmentation. '.
                format(num_instances_without_valid_segmentation) +
                'There might be issues in your dataset generation process. '
                'A valid polygon should be a list[float] with even'
                'length >= 6.')
        data_list = [v for v in videos.values()]
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        video_info = raw_data_info['raw_video_info']
        ann_info = raw_data_info['raw_ann_info']
        frame_idx = raw_data_info['frame_idx']
        data_info = {}

        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                video_info['file_names'][frame_idx])
        else:
            img_path = video_info['file_names'][frame_idx]
        data_info['img_path'] = img_path
        data_info['height'] = video_info['height']
        data_info['width'] = video_info['width']
        data_info['instances'] = ann_info

        if self.has_expression:
            data_info['task'] = 'grounding'
            data_info['dataset_name'] = self.dataset_name
        elif self.sot_task:
            data_info['task'] = 'sot'
            data_info['dataset_name'] = 'sot'
        else:
            data_info['task'] = 'detection'
            if self.dataset_name.startswith('ytvis_2019'):
                data_info['dataset_name'] = 'vis19'
            elif self.dataset_name.startswith('ytvis_2021'):
                data_info['dataset_name'] = 'vis21'
            elif self.dataset_name.startswith('ytvis_ovis'):
                data_info['dataset_name'] = 'ovis'
            elif self.dataset_name.startswith('coco'):
                data_info['dataset_name'] = 'coco'
            elif self.dataset_name.startswith('bdd_det'):
                data_info['dataset_name'] = 'bdd_det'
            elif self.dataset_name.startswith('bdd_inst'):
                data_info['dataset_name'] = 'bdd_inst'
            elif self.dataset_name.startswith(
                    'bdd_box_track') or self.dataset_name.startswith(
                        'bdd_seg_track'):
                data_info['dataset_name'] = 'bdd_track'
            else:
                raise ValueError('Unsupported dataset_name')
        return data_info

    def filter_data(self) -> List[int]:
        """Filter image annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        if self.test_mode:
            return self.data_list
        return self.data_list

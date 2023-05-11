# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict
from typing import Any, List, Tuple

import mmengine.fileio as fileio
from mmengine.dataset import BaseDataset
from mmengine.logging import print_log

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS


@DATASETS.register_module()
class BaseVideoDataset(BaseDataset):
    """Base video dataset for VID, MOT and VIS tasks."""

    META = dict(classes=None)
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self, *args, backend_args: dict = None, **kwargs):
        self.backend_args = backend_args
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        with fileio.get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the classes
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        # used in `filter_data`
        self.img_ids_with_ann = set()

        img_ids = self.coco.get_img_ids()
        total_ann_ids = []
        # if ``video_id`` is not in the annotation file, we will assign a big
        # unique video_id for this video.
        single_video_id = 100000
        videos = {}
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if 'video_id' not in raw_img_info:
                single_video_id = single_video_id + 1
                video_id = single_video_id
            else:
                video_id = raw_img_info['video_id']

            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'images': [],
                    'video_length': 0
                }

            videos[video_id]['video_length'] += 1
            ann_ids = self.coco.get_ann_ids(
                img_ids=[img_id], cat_ids=self.cat_ids)
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))

            if len(parsed_data_info['instances']) > 0:
                self.img_ids_with_ann.add(parsed_data_info['img_id'])

            videos[video_id]['images'].append(parsed_data_info)

        data_list = [v for v in videos.values()]

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            if ann.get('instance_id', None):
                instance['instance_id'] = ann['instance_id']
            else:
                # image dataset usually has no `instance_id`.
                # Therefore, we set it to `i`.
                instance['instance_id'] = i
            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[int]:
        """Filter image annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        num_imgs_before_filter = sum(
            [len(info['images']) for info in self.data_list])
        num_imgs_after_filter = 0

        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= self.img_ids_with_ann

        new_data_list = []
        for video_data_info in self.data_list:
            imgs_data_info = video_data_info['images']
            valid_imgs_data_info = []

            for data_info in imgs_data_info:
                img_id = data_info['img_id']
                width = data_info['width']
                height = data_info['height']
                # TODO: simplify these conditions
                if self.filter_cfg is None:
                    if img_id not in ids_in_cat:
                        video_data_info['video_length'] -= 1
                        continue
                    if min(width, height) >= 32:
                        valid_imgs_data_info.append(data_info)
                        num_imgs_after_filter += 1
                    else:
                        video_data_info['video_length'] -= 1
                else:
                    if self.filter_cfg.get('filter_empty_gt',
                                           True) and img_id not in ids_in_cat:
                        video_data_info['video_length'] -= 1
                        continue
                    if min(width, height) >= self.filter_cfg.get(
                            'min_size', 32):
                        valid_imgs_data_info.append(data_info)
                        num_imgs_after_filter += 1
                    else:
                        video_data_info['video_length'] -= 1
                video_data_info['images'] = valid_imgs_data_info
            new_data_list.append(video_data_info)

        print_log(
            'The number of samples before and after filtering: '
            f'{num_imgs_before_filter} / {num_imgs_after_filter}', 'current')
        return new_data_list

    def prepare_data(self, idx) -> Any:
        """Get date processed by ``self.pipeline``. Note that ``idx`` is a
        video index in default since the base element of video dataset is a
        video. However, in some cases, we need to specific both the video index
        and frame index. For example, in traing mode, we may want to sample the
        specific frames and all the frames must be sampled once in a epoch; in
        test mode, we may want to output data of a single image rather than the
        whole video for saving memory.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if isinstance(idx, tuple):
            assert len(idx) == 2, 'The length of idx must be 2: '
            '(video_index, frame_index)'
            video_idx, frame_idx = idx[0], idx[1]
        else:
            video_idx, frame_idx = idx, None

        data_info = self.get_data_info(video_idx)
        if self.test_mode:
            # Support two test_mode: frame-level and video-level
            final_data_info = defaultdict(list)
            if frame_idx is None:
                frames_idx_list = list(range(data_info['video_length']))
            else:
                frames_idx_list = [frame_idx]
            for index in frames_idx_list:
                frame_ann = data_info['images'][index]
                frame_ann['video_id'] = data_info['video_id']
                # Collate data_list (list of dict to dict of list)
                for key, value in frame_ann.items():
                    final_data_info[key].append(value)
                # copy the info in video-level into img-level
                # TODO: the value of this key is the same as that of
                # `video_length` in test mode
                final_data_info['ori_video_length'].append(
                    data_info['video_length'])

            final_data_info['video_length'] = [len(frames_idx_list)
                                               ] * len(frames_idx_list)
            return self.pipeline(final_data_info)
        else:
            # Specify `key_frame_id` for the frame sampling in the pipeline
            if frame_idx is not None:
                data_info['key_frame_id'] = frame_idx
            return self.pipeline(data_info)

    def get_cat_ids(self, index) -> List[int]:
        """Following image detection, we provide this interface function. Get
        category ids by video index and frame index.

        Args:
            index: The index of the dataset. It support two kinds of inputs:
                Tuple:
                    video_idx (int): Index of video.
                    frame_idx (int): Index of frame.
                Int: Index of video.

        Returns:
            List[int]: All categories in the image of specified video index
            and frame index.
        """
        if isinstance(index, tuple):
            assert len(
                index
            ) == 2, f'Expect the length of index is 2, but got {len(index)}'
            video_idx, frame_idx = index
            instances = self.get_data_info(
                video_idx)['images'][frame_idx]['instances']
            return [instance['bbox_label'] for instance in instances]
        else:
            cat_ids = []
            for img in self.get_data_info(index)['images']:
                for instance in img['instances']:
                    cat_ids.append(instance['bbox_label'])
            return cat_ids

    @property
    def num_all_imgs(self):
        """Get the number of all the images in this video dataset."""
        return sum(
            [len(self.get_data_info(i)['images']) for i in range(len(self))])

    def get_len_per_video(self, idx):
        """Get length of one video.

        Args:
            idx (int): Index of video.

        Returns:
            int (int): The length of the video.
        """
        return len(self.get_data_info(idx)['images'])

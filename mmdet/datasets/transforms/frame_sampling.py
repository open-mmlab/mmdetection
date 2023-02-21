# Copyright (c) OpenMMLab. All rights reserved.
import random
from collections import defaultdict
from typing import Dict, List, Optional

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class UniformSample(BaseTransform):

    def __init__(self,
                 num_ref_imgs=1,
                 frame_range=10,
                 filter_key_img=True,
                 collect_video_keys=['video_id', 'video_length']):
        self.num_ref_imgs = num_ref_imgs
        self.filter_key_img = filter_key_img
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')
        self.frame_range = frame_range
        self.collect_video_keys = collect_video_keys

    def sampling_frames(self,
                        video_length,
                        key_frame_id: Optional[int] = None):
        """"""
        if key_frame_id is None:
            key_frame_id = random.sample(list(range(video_length)), 1)[0]

        if video_length > 1:
            left = max(0, key_frame_id + self.frame_range[0])
            right = min(key_frame_id + self.frame_range[1], video_length - 1)
            frame_ids = list(range(0, video_length))

            valid_ids = frame_ids[left:right + 1]
            if self.filter_key_img and key_frame_id in valid_ids:
                valid_ids.remove(key_frame_id)
            assert len(
                valid_ids
            ) > 0, 'After filtering key frame, there are no valid frames'
            if len(valid_ids) < self.num_ref_imgs:
                valid_ids = valid_ids * self.num_ref_imgs
            ref_frame_ids = random.sample(valid_ids, self.num_ref_imgs)
        else:
            ref_frame_ids = [key_frame_id] * self.num_ref_imgs

        sampled_frames_ids = [key_frame_id] + ref_frame_ids
        sampled_frames_ids = sorted(sampled_frames_ids)

        key_frames_ind = sampled_frames_ids.index(key_frame_id)
        key_frame_flags = [False] * len(sampled_frames_ids)
        key_frame_flags[key_frames_ind] = True
        return sampled_frames_ids, key_frame_flags

    def prepare_data(self, video_infos: dict,
                     sampled_inds: List[int]) -> Dict[str, List]:
        frames_anns = video_infos['images']
        final_data_info = defaultdict(list)
        # for data in frames_anns:
        for index in sampled_inds:
            data = frames_anns[index]
            # copy the info in video-level into img-level
            for key in self.collect_video_keys:
                if key == 'video_length':
                    data['ori_video_length'] = video_infos[key]
                    data['video_length'] = len(sampled_inds)
                else:
                    data[key] = video_infos[key]
            # Collate data_list (list of dict to dict of list)
            for key, value in data.items():
                final_data_info[key].append(value)

        return final_data_info

    def transform(self, video_infos: dict) -> Optional[Dict[str, List]]:
        if 'key_frame_id' in video_infos:
            key_frame_id = video_infos['key_frame_id']
            assert isinstance(video_infos['key_frame_id'], int)
        else:
            key_frame_id = None

        (sampled_frames_ids, key_frame_flags) = self.sampling_frames(
            video_infos['video_length'], key_frame_id=key_frame_id)
        results = self.prepare_data(video_infos, sampled_frames_ids)
        results['key_frame_flags'] = key_frame_flags

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(num_ref_imgs={self.num_ref_imgs}, '
        repr_str += f'frame_range={self.frame_range}, '
        repr_str += f'filter_key_img={self.filter_key_img}, '
        repr_str += f'collect_video_keys={self.collect_video_keys})'
        return repr_str

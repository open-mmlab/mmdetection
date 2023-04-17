# Copyright (c) OpenMMLab. All rights reserved.
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BaseFrameSample(BaseTransform):
    """Directly get the key frame, no reference frames.

    Args:
        collect_video_keys (list[str]): The keys of video info to be
            collected.
    """

    def __init__(self,
                 collect_video_keys: List[str] = ['video_id', 'video_length']):
        self.collect_video_keys = collect_video_keys

    def prepare_data(self, video_infos: dict,
                     sampled_inds: List[int]) -> Dict[str, List]:
        """Prepare data for the subsequent pipeline.

        Args:
            video_infos (dict): The whole video information.
            sampled_inds (list[int]): The sampled frame indices.

        Returns:
            dict: The processed data information.
        """
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
        """Transform the video information.

        Args:
            video_infos (dict): The whole video information.

        Returns:
            dict: The data information of the key frames.
        """
        if 'key_frame_id' in video_infos:
            key_frame_id = video_infos['key_frame_id']
            assert isinstance(video_infos['key_frame_id'], int)
        else:
            key_frame_id = random.sample(
                list(range(video_infos['video_length'])), 1)[0]
        results = self.prepare_data(video_infos, [key_frame_id])

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_video_keys={self.collect_video_keys})'
        return repr_str


@TRANSFORMS.register_module()
class UniformRefFrameSample(BaseFrameSample):
    """Uniformly sample reference frames.

    Args:
        num_ref_imgs (int): Number of reference frames to be sampled.
        frame_range (int | list[int]): Range of frames to be sampled around
            key frame. If int, the range is [-frame_range, frame_range].
            Defaults to 10.
        filter_key_img (bool): Whether to filter the key frame when
            sampling reference frames. Defaults to True.
        collect_video_keys (list[str]): The keys of video info to be
            collected.
    """

    def __init__(self,
                 num_ref_imgs: int = 1,
                 frame_range: Union[int, List[int]] = 10,
                 filter_key_img: bool = True,
                 collect_video_keys: List[str] = ['video_id', 'video_length']):
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
        super().__init__(collect_video_keys=collect_video_keys)

    def sampling_frames(self, video_length: int, key_frame_id: int):
        """Sampling frames.

        Args:
            video_length (int): The length of the video.
            key_frame_id (int): The key frame id.

        Returns:
            list[int]: The sampled frame indices.
        """
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

    def transform(self, video_infos: dict) -> Optional[Dict[str, List]]:
        """Transform the video information.

        Args:
            video_infos (dict): The whole video information.

        Returns:
            dict: The data information of the sampled frames.
        """
        if 'key_frame_id' in video_infos:
            key_frame_id = video_infos['key_frame_id']
            assert isinstance(video_infos['key_frame_id'], int)
        else:
            key_frame_id = random.sample(
                list(range(video_infos['video_length'])), 1)[0]

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

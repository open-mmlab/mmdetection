# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, TrackDataSample


@TRANSFORMS.register_module()
class PackVideoInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation. All the information of images are packed to ``inputs``. All
    the information except images are packed to ``data_samples``. In order to
    get the original annotaiton and meta info, we add `instances` key into meta
    keys.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_ids': 'instances_ids',
        'positive_map': 'positive_map'
    }

    def __init__(self,
                 train_state: bool = False,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'video_id', 'video_length',
                                             'ori_video_length', 'instances',
                                             'expressions', 'scale_idx',
                                             'positive_map_label_to_token')):
        self.train_state = train_state
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images [N, C, H, W]
        if 'img' in results:
            imgs = results['img']
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.transpose(0, 3, 1, 2)
            packed_results['inputs'] = to_tensor(imgs)

        # 2. Pack InstanceData
        if self.train_state:
            assert 'gt_valid_soft_flag' in results \
                and 'gt_valid_flag' in results
            valid_list = results['gt_valid_flag']
            for valid_image in valid_list:
                if np.sum(valid_image).item() == 0:
                    return None

        assert 'img_path' in results, 'img_path must contained in the results'
        'for counting the number of images'

        num_imgs = len(results['img_path'])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                for i, gt_mask in enumerate(gt_masks_list):
                    instance_data_list[i][mapped_key] = gt_mask
            elif key == 'gt_instances_ids':
                gt_instances_ids_list = results[key]
                for i, instance_id in enumerate(gt_instances_ids_list):
                    if self.train_state:
                        instance_id[~results['gt_valid_soft_flag'][i]] = -1
                        if np.sum(results['gt_instances_ids'] != -1) == 0:
                            return None
                        else:
                            instance_data_list[i][self.mapping_table[
                                key]] = to_tensor(instance_id)
                    else:
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(instance_id)
            else:
                anns_list = results[key]
                for i, ann in enumerate(anns_list):
                    instance_data_list[i][self.mapping_table[key]] = to_tensor(
                        ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f'{key}': img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        track_data_sample.set_metainfo(dict(task=results['task'][0]))
        if 'key_frame_flags' in results:
            key_frame_flags = np.asarray(results['key_frame_flags'])
            key_frames_inds = np.where(key_frame_flags)[0].tolist()
            ref_frames_inds = np.where(~key_frame_flags)[0].tolist()
            track_data_sample.set_metainfo(
                dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(
                dict(ref_frames_inds=ref_frames_inds))

        packed_results['data_samples'] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from .det_data_sample import DetDataSample


class TrackDataSample(BaseDataElement):
    """A data structure interface of tracking task in MMDetection. It is used
    as interfaces between different components.

    This data structure can be viewd as a wrapper of multiple DetDataSample to
    some extent. Specifically, it only contains a property:
    ``video_data_samples`` which is a list of DetDataSample, each of which
    corresponds to a single frame. If you want to get the property of a single
    frame, you must first get the corresponding ``DetDataSample`` by indexing
    and then get the property of the frame, such as ``gt_instances``,
    ``pred_instances`` and so on. As for metainfo, it differs from
    ``DetDataSample`` in that each value corresponds to the metainfo key is a
    list where each element corresponds to information of a single frame.

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet.structures import DetDataSample, TrackDataSample
        >>> track_data_sample = TrackDataSample()
        >>> # set the 1st frame
        >>> frame1_data_sample = DetDataSample(metainfo=dict(
        ...         img_shape=(100, 100), frame_id=0))
        >>> frame1_gt_instances = InstanceData()
        >>> frame1_gt_instances.bbox = torch.zeros([2, 4])
        >>> frame1_data_sample.gt_instances = frame1_gt_instances
        >>> # set the 2nd frame
        >>> frame2_data_sample = DetDataSample(metainfo=dict(
        ...         img_shape=(100, 100), frame_id=1))
        >>> frame2_gt_instances = InstanceData()
        >>> frame2_gt_instances.bbox = torch.ones([3, 4])
        >>> frame2_data_sample.gt_instances = frame2_gt_instances
        >>> track_data_sample.video_data_samples = [frame1_data_sample,
        ...                                         frame2_data_sample]
        >>> # set metainfo for track_data_sample
        >>> track_data_sample.set_metainfo(dict(key_frames_inds=[0]))
        >>> track_data_sample.set_metainfo(dict(ref_frames_inds=[1]))
        >>> print(track_data_sample)
        <TrackDataSample(

            META INFORMATION
            key_frames_inds: [0]
            ref_frames_inds: [1]

            DATA FIELDS
            video_data_samples: [<DetDataSample(

                    META INFORMATION
                    img_shape: (100, 100)

                    DATA FIELDS
                    gt_instances: <InstanceData(

                            META INFORMATION

                            DATA FIELDS
                            bbox: tensor([[0., 0., 0., 0.],
                                        [0., 0., 0., 0.]])
                        ) at 0x7f639320dcd0>
                ) at 0x7f64bd223340>, <DetDataSample(

                    META INFORMATION
                    img_shape: (100, 100)

                    DATA FIELDS
                    gt_instances: <InstanceData(

                            META INFORMATION

                            DATA FIELDS
                            bbox: tensor([[1., 1., 1., 1.],
                                        [1., 1., 1., 1.],
                                        [1., 1., 1., 1.]])
                        ) at 0x7f64bd128b20>
                ) at 0x7f64bd1346d0>]
        ) at 0x7f64bd2237f0>
        >>> print(len(track_data_sample))
        2
        >>> key_data_sample = track_data_sample.get_key_frames()
        >>> print(key_data_sample[0].frame_id)
        0
        >>> ref_data_sample = track_data_sample.get_ref_frames()
        >>> print(ref_data_sample[0].frame_id)
        1
        >>> frame1_data_sample = track_data_sample[0]
        >>> print(frame1_data_sample.gt_instances.bbox)
        tensor([[0., 0., 0., 0.],
                [0., 0., 0., 0.]])
        >>> # Tensor-like methods
        >>> cuda_track_data_sample = track_data_sample.to('cuda')
        >>> cuda_track_data_sample = track_data_sample.cuda()
        >>> cpu_track_data_sample = track_data_sample.cpu()
        >>> cpu_track_data_sample = track_data_sample.to('cpu')
        >>> fp16_instances = cuda_track_data_sample.to(
        ...     device=None, dtype=torch.float16, non_blocking=False,
        ...     copy=False, memory_format=torch.preserve_format)
    """

    @property
    def video_data_samples(self) -> List[DetDataSample]:
        return self._video_data_samples

    @video_data_samples.setter
    def video_data_samples(self, value: List[DetDataSample]):
        if isinstance(value, DetDataSample):
            value = [value]
        assert isinstance(value, list), 'video_data_samples must be a list'
        assert isinstance(
            value[0], DetDataSample
        ), 'video_data_samples must be a list of DetDataSample, but got '
        f'{value[0]}'
        self.set_field(value, '_video_data_samples', dtype=list)

    @video_data_samples.deleter
    def video_data_samples(self):
        del self._video_data_samples

    def __getitem__(self, index):
        assert hasattr(self,
                       '_video_data_samples'), 'video_data_samples not set'
        return self._video_data_samples[index]

    def get_key_frames(self):
        assert hasattr(self, 'key_frames_inds'), \
            'key_frames_inds not set'
        assert isinstance(self.key_frames_inds, Sequence)
        key_frames_info = []
        for index in self.key_frames_inds:
            key_frames_info.append(self[index])
        return key_frames_info

    def get_ref_frames(self):
        assert hasattr(self, 'ref_frames_inds'), \
            'ref_frames_inds not set'
        ref_frames_info = []
        assert isinstance(self.ref_frames_inds, Sequence)
        for index in self.ref_frames_inds:
            ref_frames_info.append(self[index])
        return ref_frames_info

    def __len__(self):
        return len(self._video_data_samples) if hasattr(
            self, '_video_data_samples') else 0

    # TODO: add UT for this Tensor-like method
    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if hasattr(v, 'to'):
                    v = v.to(*args, **kwargs)
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.cpu()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.cuda()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def npu(self) -> 'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.npu()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.detach()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.detach().cpu().numpy()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                elif isinstance(v, BaseDataElement):
                    v = v.to_tensor()
                data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def clone(self) -> 'BaseDataElement':
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))

        for k, v_list in self.items():
            clone_item_list = []
            for v in v_list:
                clone_item_list.append(v.clone())
            clone_data.set_data({k: clone_item_list})
        return clone_data


TrackSampleList = List[TrackDataSample]
OptTrackSampleList = Optional[TrackSampleList]

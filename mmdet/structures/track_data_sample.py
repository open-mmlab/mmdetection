# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

from mmengine.structures import BaseDataElement

from .det_data_sample import DetDataSample


class TrackDataSample(BaseDataElement):

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
        ), 'video_data_samples must be a list of DetDataSample'
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


SampleTrackList = List[TrackDataSample]
OptSampleTrackList = Optional[SampleTrackList]

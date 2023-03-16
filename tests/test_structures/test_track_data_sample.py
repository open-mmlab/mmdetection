from unittest import TestCase

import pytest

from mmdet.structures import DetDataSample, TrackDataSample


class TestDetDataSample(TestCase):

    def test_init(self):
        track_data_sample = TrackDataSample(
            metainfo=dict(key_frames_inds=[0], ref_frames_inds=[1]))

        assert 'key_frames_inds' in track_data_sample.metainfo and \
            'ref_frames_inds' in track_data_sample.metainfo
        assert track_data_sample.key_frames_inds == [0]
        assert track_data_sample.ref_frames_inds == [1]
        with pytest.raises(AssertionError):
            track_data_sample.get_key_frames()
        with pytest.raises(AssertionError):
            track_data_sample.get_ref_frames()

    def test_setter(self):
        det_data_sample_1 = DetDataSample(
            metainfo=dict(scale_factor=(1.5, 1.5)))
        det_data_sample_2 = DetDataSample(metainfo=dict(scale_factor=(2., 2.)))
        track_data_sample = TrackDataSample(
            metainfo=dict(key_frames_inds=[0], ref_frames_inds=[1]))
        track_data_sample.video_data_samples = [
            det_data_sample_1, det_data_sample_2
        ]

        assert track_data_sample.get_key_frames()[0].scale_factor == (1.5, 1.5)
        assert track_data_sample.get_ref_frames()[0].scale_factor == (2., 2.)

    def test_deleter(self):
        det_data_sample_1 = DetDataSample(
            metainfo=dict(scale_factor=(1.5, 1.5)))
        det_data_sample_2 = DetDataSample(metainfo=dict(scale_factor=(2., 2.)))
        track_data_sample = TrackDataSample(
            metainfo=dict(key_frames_inds=[0], ref_frames_inds=[1]))
        track_data_sample.video_data_samples = [
            det_data_sample_1, det_data_sample_2
        ]
        assert 'video_data_samples' in track_data_sample
        del track_data_sample.video_data_samples
        assert 'video_data_samples' not in track_data_sample

import unittest

import numpy as np

from mmdet.datasets.transforms import BaseFrameSample, UniformRefFrameSample


class TestFrameSample(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod()
        -> tearDown() -> cleanUp()
        """
        self.H, self.W = 5, 8
        self.img = np.zeros((self.H, self.W, 3))
        self.gt_bboxes = np.zeros((2, 4))
        self.gt_bboxes_labels = [
            np.zeros((2, )),
            np.zeros((2, )) + 1,
            np.zeros((2, )) - 1
        ]
        self.gt_instances_id = [
            np.ones((2, ), dtype=np.int32),
            np.ones((2, ), dtype=np.int32) - 1,
            np.ones((2, ), dtype=np.int32) + 1
        ]
        self.frame_id = [0, 1, 2]
        self.scale_factor = [1.0, 1.5, 2.0]
        self.flip = [False] * 3
        self.ori_shape = [(self.H, self.W)] * 3
        self.img_id = [0, 1, 2]

        self.video_infos = dict(video_id=0, video_length=10, key_frame_id=4)
        self.video_infos['images'] = []
        self.info_keys = [
            'video_id', 'video_length', 'img', 'gt_bboxes', 'gt_bboxes_labels',
            'gt_instances_id', 'img_id', 'frame_id'
        ]
        for i in range(10):
            frame_info = dict(
                img=np.zeros((self.H, self.W, 3)) + i,
                gt_bboxes=np.zeros((2, 4)) + i,
                gt_bboxes_labels=np.zeros((2, )) + i,
                gt_instances_id=np.zeros((2, ), dtype=np.int32) + i,
                ori_shape=(self.H + i, self.W + i),
                frame_id=i,
                img_id=i)
            self.video_infos['images'].append(frame_info)

    def test_base_frame_sample(self):
        sampler = BaseFrameSample()
        results = sampler(self.video_infos)
        assert isinstance(results, dict)
        for key in self.info_keys:
            assert key in results
            assert len(results[key]) == 1
            if key == 'frame_id':
                assert results[key] == [4]

        key_frame_id = self.video_infos['key_frame_id']
        assert (results['img'][0] == np.zeros(
            (self.H, self.W, 3)) + key_frame_id).all()
        assert (results['gt_bboxes'][0] == np.zeros(
            (2, 4)) + key_frame_id).all()
        assert (results['gt_bboxes_labels'][0] == np.zeros(
            (2, )) + key_frame_id).all()
        assert (results['gt_instances_id'][0] == np.zeros(
            (2, )) + key_frame_id).all()
        assert results['ori_shape'][0] == (self.H + key_frame_id,
                                           self.W + key_frame_id)
        assert results['img_id'][0] == key_frame_id

    def test_uniform_ref_frame_sample(self):
        sampler = UniformRefFrameSample(
            num_ref_imgs=2, frame_range=[-1, 1], filter_key_img=True)
        results = sampler(self.video_infos)
        assert isinstance(results, dict)
        for key in self.info_keys:
            assert key in results
            assert len(results[key]) == 3
            if key == 'frame_id':
                assert results[key] == [3, 4, 5]

        key_frame_id = self.video_infos['key_frame_id']
        assert (results['img'][1] == np.zeros(
            (self.H, self.W, 3)) + key_frame_id).all()
        assert (results['gt_bboxes'][1] == np.zeros(
            (2, 4)) + key_frame_id).all()
        assert (results['gt_bboxes_labels'][1] == np.zeros(
            (2, )) + key_frame_id).all()
        assert (results['gt_instances_id'][1] == np.zeros(
            (2, )) + key_frame_id).all()
        assert results['ori_shape'][1] == (self.H + key_frame_id,
                                           self.W + key_frame_id)
        assert results['img_id'][1] == key_frame_id

        # test the filter_key_img and the correctness of returned frame index
        sampler = UniformRefFrameSample(
            num_ref_imgs=2, frame_range=[0, 1], filter_key_img=False)
        results = sampler(self.video_infos)
        assert 4 in results['img_id'] and results['img_id'].count(4) == 2
        assert 5 in results['img_id'] and results['img_id'].count(5) == 1
        assert results['key_frame_flags'] == [True, False, False]

    def test_repr(self):
        transform = BaseFrameSample()
        self.assertEqual(
            repr(transform),
            "BaseFrameSample(collect_video_keys=['video_id', 'video_length'])")

        transform = UniformRefFrameSample(
            num_ref_imgs=2, frame_range=10, filter_key_img=True)
        self.assertEqual(
            repr(transform),
            ('UniformRefFrameSample(num_ref_imgs=2, '
             'frame_range=[-10, 10], filter_key_img=True, '
             "collect_video_keys=['video_id', 'video_length'])"))

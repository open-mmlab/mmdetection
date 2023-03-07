# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.evaluator import Evaluator
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.testing import VideoDataSampleFeeder
from mmdet.utils import register_all_modules

register_all_modules()


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        labels = torch.stack(data_samples)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class ToyModel1(BaseModel, ToyModel):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return super(BaseModel, self).forward(*args, **kwargs)


class ToyModel2(BaseModel):

    def __init__(self):
        super().__init__()
        self.teacher = ToyModel1()
        self.student = ToyModel1()
        self.semi_test_cfg = dict(predict_on='teacher')

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)


@DATASETS.register_module(force=True)
class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


class TestTeacherStudentValLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_teacher_student_val_loop(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel2().to(device)
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        evaluator.__class__ = Evaluator
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(type='TeacherStudentValLoop'),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            experiment_name='test1')
        runner.train()


class TestVideoValLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_iter_video(self):
        video_length = 3
        track_data_sample = TrackDataSample(
            metainfo=dict(ori_video_length=video_length))
        video_data_samples = []
        for i in range(video_length):
            det_data_sample = DetDataSample(metainfo=dict(frame_id=i))
            video_data_samples.append(det_data_sample)
        track_data_sample.video_data_samples = video_data_samples
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel1().to(device)
        model.val_step = Mock(return_value=[track_data_sample])
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        evaluator.__class__ = Evaluator
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=2,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(type='VideoValLoop'),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            experiment_name='val_video_test')

        runner.train()
        evaluator_input = evaluator.mock_calls[0].kwargs['data_samples']
        assert isinstance(evaluator_input, list)
        assert evaluator_input[0].ori_video_length == video_length
        assert len(evaluator_input[0]) == video_length

    def test_run_iter_img(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        video_length = 3
        model = ToyModel1().to(device)

        model.val_step = VideoDataSampleFeeder(video_length)
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        evaluator.__class__ = Evaluator
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=2,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(type='VideoValLoop'),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            experiment_name='val_img_test')

        runner.train()
        evaluator_input = evaluator.mock_calls[0].kwargs['data_samples']
        assert isinstance(evaluator_input, list)
        assert len(evaluator_input[0]) == video_length
        assert evaluator_input[0].ori_video_length == video_length
        assert evaluator_input[0].video_length == video_length


class TestVideoTestLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_iter_video(self):
        video_length = 3
        track_data_sample = TrackDataSample(
            metainfo=dict(ori_video_length=video_length))
        video_data_samples = []
        for i in range(video_length):
            det_data_sample = DetDataSample(metainfo=dict(frame_id=i))
            video_data_samples.append(det_data_sample)
        track_data_sample.video_data_samples = video_data_samples
        model = ToyModel()
        model.test_step = Mock(return_value=[track_data_sample])
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        evaluator.__class__ = Evaluator
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            test_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            test_cfg=dict(type='VideoTestLoop'),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            experiment_name='test_video_test')

        runner.test()
        evaluator_input = evaluator.mock_calls[0].kwargs['data_samples']
        assert isinstance(evaluator_input, list)
        assert evaluator_input[0].ori_video_length == video_length
        assert len(evaluator_input[0]) == video_length

    def test_run_iter_img(self):
        video_length = 3
        model = ToyModel()
        model.test_step = VideoDataSampleFeeder(video_length)
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        evaluator.__class__ = Evaluator
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            test_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            test_cfg=dict(type='VideoTestLoop'),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            experiment_name='test_img_test')

        runner.test()
        evaluator_input = evaluator.mock_calls[0].kwargs['data_samples']
        assert isinstance(evaluator_input, list)
        assert len(evaluator_input[0]) == video_length
        assert evaluator_input[0].ori_video_length == video_length
        assert evaluator_input[0].video_length == video_length

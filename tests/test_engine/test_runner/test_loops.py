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

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
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


class ToyMetric1(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestMeanTeacherHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mean_teacher_hook(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel2().to(device)
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=[ToyMetric1()],
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='MeanTeacherHook')],
            experiment_name='test1')
        runner.train()

        self.assertTrue(
            osp.exists(osp.join(self.temp_dir.name, 'epoch_2.pth')))
        # checkpoint = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))

        # load and testing
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=[ToyMetric1()],
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='MeanTeacherHook')],
            experiment_name='test2')
        runner.test()

        @MODEL_WRAPPERS.register_module()
        class DummyWrapper(BaseModel):

            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        # with model wrapper
        runner = Runner(
            model=DummyWrapper(ToyModel2()),
            test_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=[ToyMetric1()],
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            default_scope='mmdet',
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='MeanTeacherHook')],
            experiment_name='test3')
        runner.test()

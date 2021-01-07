import os.path as osp
import tempfile
import unittest.mock as mock
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmdet.core import EvalHook, SWAHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.eval_result = [0.1, 0.4, 0.3, 0.7, 0.2, 0.05, 0.4, 0.6]

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1]))
        return results

    def __len__(self):
        return 1

    @mock.create_autospec
    def evaluate(self, results, logger=None):
        pass


class EvalDataset(ExampleDataset):

    def evaluate(self, results, logger=None):
        mean_ap = self.eval_result[self.index]
        output = OrderedDict(mAP=mean_ap, index=self.index, score=mean_ap)
        self.index += 1
        return output


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, rescale=False, return_loss=False):
        return imgs

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {
            'loss': 0.5,
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 1
        }
        return outputs


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@patch('mmdet.apis.single_gpu_test', MagicMock)
@patch('mmdet.apis.multi_gpu_test', MagicMock)
def test_swa_hook():
    with pytest.raises(TypeError):
        # swa_eval must be of type bool
        test_dataset = ExampleDataset()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        eval_hook = EvalHook(data_loader)
        swa_eval = None
        SWAHook(swa_eval, eval_hook)

    with pytest.raises(TypeError):
        # eval_hook must be of either EvalHook or DistEvalHook
        # when swa_eval = True
        eval_hook = None
        swa_eval = True
        SWAHook(swa_eval, eval_hook)

    # the model should be saved as swa_model_1.pth
    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    data_loader = DataLoader(test_dataset, batch_size=1)
    eval_hook = EvalHook(data_loader, save_best=None)
    swa_eval = True
    swa_hook = SWAHook(swa_eval, eval_hook)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)

        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(swa_hook)
        runner.run([loader], [('train', 1)], 1)
        swa_model_path = osp.join(tmpdir, 'swa_model_1.pth')
        assert osp.exists(swa_model_path)

    # swa_hook should be able to evaluate the model
    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(data_loader, interval=1, save_best='auto')
    swa_eval = True
    swa_hook = SWAHook(swa_eval, eval_hook)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(swa_hook)
        runner.run([loader], [('train', 1)], 8)
        assert swa_hook.meta['hook_msgs']['best_score'] == 0.7
import os.path as osp
import tempfile
import unittest.mock as mock
from collections import OrderedDict

import mmcv
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmdet.core import EvalHook


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
def test_eval_hook():
    with pytest.raises(TypeError):
        # dataloader must be a pytorch DataLoader
        test_dataset = ExampleModel()
        data_loader = [
            DataLoader(
                test_dataset,
                batch_size=1,
                sampler=None,
                num_worker=0,
                shuffle=False)
        ]
        EvalHook(data_loader)

    with pytest.raises(KeyError):
        # rule must be in keys of rule_map
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EvalHook(data_loader, save_best=True, rule='unsupport')

    with pytest.raises(ValueError):
        # key_indicator must be valid when rule_map is None
        test_dataset = ExampleModel()
        data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=None,
            num_workers=0,
            shuffle=False)
        EvalHook(data_loader, save_best=True, key_indicator='unsupport')

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)

    data_loader = DataLoader(test_dataset, batch_size=1)
    eval_hook = EvalHook(data_loader, save_best=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 1)
        test_dataset.evaluate.assert_called_with(
            test_dataset, [torch.tensor([1])], logger=runner.logger)

        best_json_path = osp.join(tmpdir, 'best.json')
        assert not osp.exists(best_json_path)

    # when `save_best` is True, `key_indicator` is not set, first metric
    # should be used.
    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(data_loader, interval=1, save_best=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.7
        assert best_json['key_indicator'] == 'mAP'

    loader = DataLoader(EvalDataset(), batch_size=1)
    model = ExampleModel()
    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(
        data_loader, interval=1, save_best=True, key_indicator='mAP')

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.7
        assert best_json['key_indicator'] == 'mAP'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(
        data_loader,
        interval=1,
        save_best=True,
        key_indicator='score',
        rule='greater')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.7
        assert best_json['key_indicator'] == 'score'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(
        data_loader, save_best=True, rule='less', key_indicator='mAP')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_6.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.05
        assert best_json['key_indicator'] == 'mAP'

    data_loader = DataLoader(EvalDataset(), batch_size=1)
    eval_hook = EvalHook(data_loader, save_best=True, key_indicator='mAP')
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = get_logger('test_eval')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.run([loader], [('train', 1)], 2)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_2.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.4
        assert best_json['key_indicator'] == 'mAP'

        resume_from = osp.join(tmpdir, 'latest.pth')
        loader = DataLoader(ExampleDataset(), batch_size=1)
        eval_hook = EvalHook(data_loader, save_best=True, key_indicator='mAP')
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook)
        runner.resume(resume_from)
        runner.run([loader], [('train', 1)], 8)

        best_json_path = osp.join(tmpdir, 'best.json')
        best_json = mmcv.load(best_json_path)
        real_path = osp.join(tmpdir, 'epoch_4.pth')

        assert best_json['best_ckpt'] == osp.realpath(real_path)
        assert best_json['best_score'] == 0.7
        assert best_json['key_indicator'] == 'mAP'

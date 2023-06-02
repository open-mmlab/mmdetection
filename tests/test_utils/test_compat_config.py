import pytest
from mmcv import ConfigDict

from mmdet.utils.compat_config import (compat_imgs_per_gpu, compat_loader_args,
                                       compat_runner_args)


def test_compat_runner_args():
    cfg = ConfigDict(dict(total_epochs=12))
    with pytest.warns(None) as record:
        cfg = compat_runner_args(cfg)
    assert len(record) == 1
    assert 'runner' in record.list[0].message.args[0]
    assert 'runner' in cfg
    assert cfg.runner.type == 'EpochBasedRunner'
    assert cfg.runner.max_epochs == cfg.total_epochs


def test_compat_loader_args():
    cfg = ConfigDict(dict(data=dict(val=dict(), test=dict(), train=dict())))
    cfg = compat_loader_args(cfg)
    # auto fill loader args
    assert 'val_dataloader' in cfg.data
    assert 'train_dataloader' in cfg.data
    assert 'test_dataloader' in cfg.data
    cfg = ConfigDict(
        dict(
            data=dict(
                samples_per_gpu=1,
                persistent_workers=True,
                workers_per_gpu=1,
                val=dict(samples_per_gpu=3),
                test=dict(samples_per_gpu=2),
                train=dict())))

    cfg = compat_loader_args(cfg)

    assert cfg.data.train_dataloader.workers_per_gpu == 1
    assert cfg.data.train_dataloader.samples_per_gpu == 1
    assert cfg.data.train_dataloader.persistent_workers
    assert cfg.data.val_dataloader.workers_per_gpu == 1
    assert cfg.data.val_dataloader.samples_per_gpu == 3
    assert cfg.data.test_dataloader.workers_per_gpu == 1
    assert cfg.data.test_dataloader.samples_per_gpu == 2

    # test test is a list
    cfg = ConfigDict(
        dict(
            data=dict(
                samples_per_gpu=1,
                persistent_workers=True,
                workers_per_gpu=1,
                val=dict(samples_per_gpu=3),
                test=[dict(samples_per_gpu=2),
                      dict(samples_per_gpu=3)],
                train=dict())))

    cfg = compat_loader_args(cfg)
    assert cfg.data.test_dataloader.samples_per_gpu == 3

    # assert can not set args at the same time
    cfg = ConfigDict(
        dict(
            data=dict(
                samples_per_gpu=1,
                persistent_workers=True,
                workers_per_gpu=1,
                val=dict(samples_per_gpu=3),
                test=dict(samples_per_gpu=2),
                train=dict(),
                train_dataloader=dict(samples_per_gpu=2))))
    # samples_per_gpu can not be set in `train_dataloader`
    # and data field at the same time
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)
    cfg = ConfigDict(
        dict(
            data=dict(
                samples_per_gpu=1,
                persistent_workers=True,
                workers_per_gpu=1,
                val=dict(samples_per_gpu=3),
                test=dict(samples_per_gpu=2),
                train=dict(),
                val_dataloader=dict(samples_per_gpu=2))))
    # samples_per_gpu can not be set in `val_dataloader`
    # and data field at the same time
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)
    cfg = ConfigDict(
        dict(
            data=dict(
                samples_per_gpu=1,
                persistent_workers=True,
                workers_per_gpu=1,
                val=dict(samples_per_gpu=3),
                test=dict(samples_per_gpu=2),
                test_dataloader=dict(samples_per_gpu=2))))
    # samples_per_gpu can not be set in `test_dataloader`
    # and data field at the same time
    with pytest.raises(AssertionError):
        compat_loader_args(cfg)


def test_compat_imgs_per_gpu():
    cfg = ConfigDict(
        dict(
            data=dict(
                imgs_per_gpu=1,
                samples_per_gpu=2,
                val=dict(),
                test=dict(),
                train=dict())))
    cfg = compat_imgs_per_gpu(cfg)
    assert cfg.data.samples_per_gpu == cfg.data.imgs_per_gpu

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

from mmcv import ConfigDict


def compat_cfg(cfg):
    """该函数会修改一些参数以保持配置的兼容性.

    例如,它会将一些将被弃用的 参数 移动到正确的字段.
    """
    cfg = copy.deepcopy(cfg)
    cfg = compat_imgs_per_gpu(cfg)  # 如果存在imgs_per_gpu字段则将其值赋给samples_per_gpu
    cfg = compat_loader_args(cfg)   # 生成 train/val/test_dataloader字段,并将已有的一些参数赋予它们
    cfg = compat_runner_args(cfg)   # 确保runner字段存在并且它的max_epochs与total_epochs一致
    return cfg


def compat_runner_args(cfg):
    if 'runner' not in cfg:
        cfg.runner = ConfigDict({
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        })
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    return cfg


def compat_imgs_per_gpu(cfg):
    cfg = copy.deepcopy(cfg)
    if 'imgs_per_gpu' in cfg.data:
        warnings.warn('"imgs_per_gpu" 在 MMDet V2.0 中已弃用.' '请改用 "samples_per_gpu"')
        if 'samples_per_gpu' in cfg.data:
            warnings.warn(
                f'得到两个参数 "imgs_per_gpu"={cfg.data.imgs_per_gpu} 和 '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, 本实验使用"imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu}')
        else:
            warnings.warn(f'在这个实验中自动设置"samples_per_gpu"="imgs_per_gpu"={cfg.data.imgs_per_gpu}')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    return cfg


def compat_loader_args(cfg):
    """Deprecated sample_per_gpu in cfg.data."""

    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = ConfigDict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = ConfigDict()
    if 'test_dataloader' not in cfg.data:
        cfg.data['test_dataloader'] = ConfigDict()

    # special process for train_dataloader
    if 'samples_per_gpu' in cfg.data:

        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        assert 'samples_per_gpu' not in \
               cfg.data.train_dataloader, ('`samples_per_gpu` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu
        cfg.data.test_dataloader['samples_per_gpu'] = samples_per_gpu
        cfg.data.val_dataloader['samples_per_gpu'] = samples_per_gpu
    if 'persistent_workers' in cfg.data:

        persistent_workers = cfg.data.pop('persistent_workers')
        assert 'persistent_workers' not in \
               cfg.data.train_dataloader, ('`persistent_workers` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['persistent_workers'] = persistent_workers

    if 'workers_per_gpu' in cfg.data:

        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu

    # 将val/test模式下中需要指定的bs改为默认为data中的bs,由下面两个代码块被注释以及上面的
    # cfg.data.train_dataloader/test_dataloader/val_dataloader改动可以看出
    # special process for val_dataloader
    # if 'samples_per_gpu' in cfg.data.val:
    #     # keep default value of `sample_per_gpu` is 1
    #     assert 'samples_per_gpu' not in \
    #            cfg.data.val_dataloader, ('`samples_per_gpu` are set '
    #                                      'in `data.val` field and ` '
    #                                      'data.val_dataloader` at '
    #                                      'the same time. '
    #                                      'Please only set it in '
    #                                      '`data.val_dataloader`. ')
    #     cfg.data.val_dataloader['samples_per_gpu'] = \
    #         cfg.data.val.pop('samples_per_gpu')
    # special process for val_dataloader

    # in case the test dataset is concatenated
    # if isinstance(cfg.data.test, dict):
    #     if 'samples_per_gpu' in cfg.data.test:
    #         assert 'samples_per_gpu' not in \
    #                cfg.data.test_dataloader, ('`samples_per_gpu` are set '
    #                                           'in `data.test` field and ` '
    #                                           'data.test_dataloader` '
    #                                           'at the same time. '
    #                                           'Please only set it in '
    #                                           '`data.test_dataloader`. ')
    #
    #         cfg.data.test_dataloader['samples_per_gpu'] = \
    #             cfg.data.test.pop('samples_per_gpu')

    if isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            if 'samples_per_gpu' in ds_cfg:
                assert 'samples_per_gpu' not in \
                       cfg.data.test_dataloader, ('`samples_per_gpu` are set '
                                                  'in `data.test` field and ` '
                                                  'data.test_dataloader` at'
                                                  ' the same time. '
                                                  'Please only set it in '
                                                  '`data.test_dataloader`. ')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        cfg.data.test_dataloader['samples_per_gpu'] = samples_per_gpu

    return cfg

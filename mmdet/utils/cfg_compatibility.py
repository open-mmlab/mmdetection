# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings


def cfg_compatibility(cfg):
    """This function would modify some filed to keep the compatibility of
    config.

    For example, it will move some args which will be deprecated to the correct
    fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg = imgs_per_gpu_rename_compatibility(cfg)
    cfg = loader_args_compatibility(cfg)
    cfg = runner_args_compatibility(cfg)
    return cfg


def runner_args_compatibility(cfg):
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    return cfg


def imgs_per_gpu_rename_compatibility(cfg):
    cfg = copy.deepcopy(cfg)
    if 'imgs_per_gpu' in cfg.data:
        warnings.warn('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                      'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            warnings.warn(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            warnings.warn('Automatically set "samples_per_gpu"="imgs_per_gpu"='
                          f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    return cfg


def loader_args_compatibility(cfg):
    """Deprecated sample_per_gpu in cfg.data."""

    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = dict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = dict()
    if 'test_dataloader' not in cfg.data:
        cfg.data['test_dataloader'] = dict()

    # special process for train_dataloader
    if 'samples_per_gpu' in cfg.data:
        warnings.warn('`samples_per_gpu` in '
                      'data will be deprecated, you should'
                      ' move it to corresponding '
                      '`*_dataloader` field')
        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        assert 'samples_per_gpu' not in \
               cfg.data.train_dataloader, 'You set ' \
                                          '`samples_per_gpu` in data field' \
                                          'and `train_dataloader` field at ' \
                                          'the same time. ' \
                                          'please only set it in ' \
                                          '`train_dataloader`'
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu

    if 'workers_per_gpu' in cfg.data:
        warnings.warn('`workers_per_gpu` in '
                      'data will be deprecated, you should'
                      ' move it to corresponding '
                      '`*_dataloader` field')
        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu

    # special process for val_dataloader
    if 'samples_per_gpu' in cfg.data.val:
        warnings.warn('`samples_per_gpu` in `val` field of '
                      'data will be deprecated, you should'
                      ' move it to `val_dataloader` field')
        # keep default value of `sample_per_gpu` is 1
        assert 'samples_per_gpu' not in \
               cfg.data.val_dataloader, 'You set ' \
               '`samples_per_gpu` in data field' \
               'and `val_dataloader` field at the' \
               'same time  ' \
               'please only set it in ' \
               '`val_dataloader`'
        cfg.data.val_dataloader['samples_per_gpu'] = \
            cfg.data.val.pop('samples_per_gpu')
    # special process for val_dataloader

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            cfg.data.test_dataloader['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')

    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        cfg.data.test_dataloader['samples_per_gpu'] = samples_per_gpu

    return cfg

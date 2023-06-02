# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import logging
import os
import platform
import warnings

import cv2
import torch.multiprocessing as mp
from mmengine import DefaultScope
from mmengine.logging import print_log
from mmengine.utils import digit_version


def setup_cache_size_limit_of_dynamo():
    """Setup cache size limit of dynamo.

    Note: Due to the dynamic shape of the loss calculation and
    post-processing parts in the object detection algorithm, these
    functions must be compiled every time they are run.
    Setting a large value for torch._dynamo.config.cache_size_limit
    may result in repeated compilation, which can slow down training
    and testing speed. Therefore, we need to set the default value of
    cache_size_limit smaller. An empirical value is 4.
    """

    import torch
    if digit_version(torch.__version__) >= digit_version('2.0.0'):
        if 'DYNAMO_CACHE_SIZE_LIMIT' in os.environ:
            import torch._dynamo
            cache_size_limit = int(os.environ['DYNAMO_CACHE_SIZE_LIMIT'])
            torch._dynamo.config.cache_size_limit = cache_size_limit
            print_log(
                f'torch._dynamo.config.cache_size_limit is force '
                f'set to {cache_size_limit}.',
                logger='current',
                level=logging.WARNING)


def setup_multi_processes(cfg):
    """设置多进程环境变量."""
    # 将多进程启动方法设置为"fork"以加快训练速度
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'多处理启动方法`{mp_start_method}`不同于之前的设置`{current_method}`.'
                f'它将被强制设置为 `{mp_start_method}`. 你可以通过更改配置文件中的"mp_start_method"来更改此行为.')
        mp.set_start_method(mp_start_method, force=True)

    # 禁用opencv多线程以避免系统过载
    opencv_num_threads = cfg.get('opencv_num_threads', 0)  # 默认为0
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # 此代码引用自 https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)  # 默认为2
    if 'train_dataloader' in cfg.data:  # 一般不存在
        workers_per_gpu = \
            max(cfg.data.train_dataloader.get('workers_per_gpu', 1),
                workers_per_gpu)

    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1  # 可参考 https://github.com/pytorch/pytorch/pull/22501#issuecomment-509966845
        warnings.warn(
            f'将每个进程的 OMP_NUM_THREADS 环境变量默认设置为 {omp_num_threads},'
            f'为避免你的系统过载,请根据需要进一步调整变量以获得最佳应用程序的性能.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # 设置 MKL线程数量
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'将每个进程的 MKL_NUM_THREADS 环境变量默认设置为 {mkl_num_threads},'
            f'为避免你的系统过载,请根据需要进一步调整变量以获得最佳应用程序的性能.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmdet into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmdet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmdet`, and all registries will build modules from mmdet's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmdet.datasets  # noqa: F401,F403
    import mmdet.engine  # noqa: F401,F403
    import mmdet.evaluation  # noqa: F401,F403
    import mmdet.models  # noqa: F401,F403
    import mmdet.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmdet')
        if never_created:
            DefaultScope.get_instance('mmdet', scope_name='mmdet')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmdet':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmdet", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmdet". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmdet-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmdet')

# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import warnings

import cv2
import torch.multiprocessing as mp


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

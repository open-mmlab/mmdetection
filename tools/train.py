# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='模型配置路径')
    parser.add_argument('--work-dir', help='存储日志和模型的目录')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='启用自动混合精度训练')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='自动缩放学习率.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='如果指定权重路径,则从它恢复训练.而如果没有指定,则尝试从work-dir中'
             '最近一次保存的模型恢复训练.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置中的一些设置, 通过键值对的方式xxx=yyy. 如果被覆盖的值是一个列表,'
             '它应该像 key="[a,b]" 或 key=a,b 格式它也允许嵌套的list tuple值,'
             '例如key="[(a,b),(c,d)]" 注意引号是必须的,不能有空格.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='分布式训练使用的启动器')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:  # 设置默认 LOCAL_RANK为当前的 local_rank
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()  # 获得命令行参数

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # 读取配置文件
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:  # 更新cfg_options中的配置信息到cfg中去
        cfg.merge_from_dict(args.cfg_options)

    # work_dir的命名优先级: 命令行 > cfg.work_dir > config文件名
    if args.work_dir is not None:
        # 如果 args.work_dir 非空,则根据"命令行参数"更新配置
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果 cfg.work_dir 为 None,则使用 ./work_dirs/cfg文件名 作为默认工作目录
        cfg.work_dir = osp.join('./work_dirs',  # basename->获取文件全名 splitext->分割文件名与后缀
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()

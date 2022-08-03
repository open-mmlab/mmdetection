# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='模型文件路径', default=r'D:\mmdetection\configs\retinanet\retinanet_r50_fpn_1x_coco.py')
    parser.add_argument('--work-dir', help='存储日志和模型的目录')
    parser.add_argument('--resume-from', help='恢复训练的目录')
    parser.add_argument('--auto-resume', action='store_true', help='自动从最近的保存点恢复训练')
    parser.add_argument('--no-validate', action='store_true', help='训练时是否不进行验证')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU的索引(仅适用于非分布式训练)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true', help='是否为不同的ranks设置不同的种子')
    parser.add_argument('--deterministic',action='store_true',help='是否固定CUDNN后端算法.以减少网络输出的随机性')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置中的一些设置, 通过键值对的方式'
                        'xxx=yyy. 如果被覆盖的值是一个列表,它应该像 key="[a,b]" 或 key=a,b 格式'
                        '它也允许嵌套的list tuple值,例如key="[(a,b),(c,d)]" 注意引号是必须的,不能有空格.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='分布式训练使用的启动器')
    parser.add_argument('--local_rank', type=int, default=0)  # 本地进程编号,此参数 torch.distributed.launch 会自动传入
    parser.add_argument('--auto-scale-lr', action='store_false', help='自动缩放学习率.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:  # 如果环境中没有 LOCAL_RANK,就设置它为当前的 local_rank
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()  # 获得命令行参数

    cfg = Config.fromfile(args.config)  # 读取配置文件

    # replace the ${key} with the value of cfg.key 目前还不清楚有什么用处
    cfg = replace_cfg_vals(cfg)

    # 更新cfg.data_root,如果MMDET_DATASETS存在环境变量中
    update_data_root(cfg)

    if args.cfg_options is not None:  # 更新cfg_options中的配置信息到cfg中去
        cfg.merge_from_dict(args.cfg_options)

    if args.auto_scale_lr:  # 如果开启自适应学习率,那么cfg文件中也必须存在相关该参数
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('在您的配置文件中找不到"auto_scale_lr"或"auto_scale_lr.enable"'
                          '或"auto_scale_lr.base_batch_size".请将所有配置文件更新到 mmdet >= 2.24.1')

    # 设置多进程配置
    setup_multi_processes(cfg)

    # 设置 cudnn_benchmark 在那些输入固定的模型中(比如SSD300),开启该参数后训练会更快
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir的命名优先级: 命令行 > cfg.work_dir > cfg文件名
    if args.work_dir is not None:
        # 如果 args.work_dir 非空,则根据"命令行参数"更新配置
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果 cfg.work_dir 为 None,则使用 ./work_dirs/cfg文件名 作为默认工作目录
        cfg.work_dir = osp.join('./work_dirs',  # basename->获取文件全名 splitext->分割文件名与后缀
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:    # 如果命令行上的resume_from非空,则更新到cfg中
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume  # 将args中的auto_resume更新到cfg中
    cfg.gpu_ids = [args.gpu_id]         # 将gou_id 由int->list 兼容分布式训练

    # 首先初始化分布式环境, 因为 logger 会依赖分布式信息.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # 通过分布式训练模式重置gpu_ids
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))  # 相对目录 -> 绝对目录 如果该目录不存在则创建否则跳过
    # 将配置文件保存到绝对目录下,此时保存的cfg是结合args参数后的而非原始cfg
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # 在其他步骤之前初始化 logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')  # 生成以时间戳为格式的日志文件
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)  # 获取logger对象,只有主进程且传入log_file才保存输出

    # 初始化meta dict来记录环境信息、种子数等一些重要信息
    meta = dict()
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # 记录一些基本信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()  # cpu, cuda or mlu
    # 设置随机种子数
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed  # 是否让不同进程的训练任务的seed不同,默认所有进程seed一致
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # 将mmdet版本、配置文件内容和识别类名作为为元数据保存到权重中
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # 为model添加CLASSES属性以方便可视化
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

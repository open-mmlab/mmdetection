# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from argparse import ArgumentParser

from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger, print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.testing import replace_to_ceph
from mmdet.utils import register_all_modules, replace_cfg_vals


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--ceph', action='store_true')
    parser.add_argument('--save-ckpt', action='store_true')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


# TODO: Need to refactor train.py so that it can be reused.
def fast_train_model(config_name, args, logger=None):
    cfg = Config.fromfile(config_name)
    cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join(args.work_dir,
                                osp.splitext(osp.basename(config_name))[0])
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config_name))[0])

    ckpt_hook = cfg.default_hooks.checkpoint
    by_epoch = ckpt_hook.get('by_epoch', True)
    fast_stop_hook = dict(type='FastStopTrainingHook')
    fast_stop_hook['by_epoch'] = by_epoch
    if args.save_ckpt:
        if by_epoch:
            interval = 1
            stop_iter_or_epoch = 2
        else:
            interval = 4
            stop_iter_or_epoch = 10
        fast_stop_hook['stop_iter_or_epoch'] = stop_iter_or_epoch
        fast_stop_hook['save_ckpt'] = True
        ckpt_hook.interval = interval

    if 'custom_hooks' in cfg:
        cfg.custom_hooks.append(fast_stop_hook)
    else:
        custom_hooks = [fast_stop_hook]
        cfg.custom_hooks = custom_hooks

    # TODO: temporary plan
    if 'visualizer' in cfg:
        if 'name' in cfg.visualizer:
            del cfg.visualizer.name

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

    if args.ceph:
        replace_to_ceph(cfg)

    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.train()


# Sample test whether the train code is correct
def main(args):
    # register all modules in mmdet into the registries
    register_all_modules(init_default_scope=False)

    config = Config.fromfile(args.config)

    # test all model
    logger = MMLogger.get_instance(
        name='MMLogger',
        log_file='benchmark_train.log',
        log_level=logging.ERROR)

    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'], flush=True)
            config_name = model_info['config'].strip()
            try:
                fast_train_model(config_name, args, logger)
            except RuntimeError as e:
                # quick exit is the normal exit message
                if 'quick exit' not in repr(e):
                    logger.error(f'{config_name} " : {repr(e)}')
            except Exception as e:
                logger.error(f'{config_name} " : {repr(e)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from argparse import ArgumentParser

from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.testing import replace_to_ceph
from mmdet.utils import register_all_modules, replace_cfg_vals


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path')
    parser.add_argument('--work-dir', help='the dir to save logs')
    parser.add_argument('--ceph', action='store_true')
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


# TODO: Need to refactor test.py so that it can be reused.
def fast_test_model(config_name, checkpoint, args, logger=None):
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

    if args.ceph:
        replace_to_ceph(cfg)

    cfg.load_from = checkpoint

    # TODO: temporary plan
    if 'visualizer' in cfg:
        if 'name' in cfg.visualizer:
            del cfg.visualizer.name

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.test()


# Sample test whether the inference code is correct
def main(args):
    # register all modules in mmdet into the registries
    register_all_modules(init_default_scope=False)

    config = Config.fromfile(args.config)

    # test all model
    logger = MMLogger.get_instance(
        name='MMLogger',
        log_file='benchmark_test.log',
        log_level=logging.ERROR)

    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'], flush=True)
            config_name = model_info['config'].strip()
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info['checkpoint'].strip())
            try:
                fast_test_model(config_name, checkpoint, args, logger)
            except Exception as e:
                logger.error(f'{config_name} " : {repr(e)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

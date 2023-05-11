# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from itertools import product

from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger, print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet tracking test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--detector', help='detection checkpoint file')
    parser.add_argument('--reid', help='reid checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
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
    return args


def get_search_params(cfg, search_params=None, prefix=None, logger=None):
    if search_params is None:
        search_params = dict()
    for k, v in cfg.items():
        if prefix is not None:
            entire_k = prefix + '.' + k
        else:
            entire_k = k
        if isinstance(v, list):
            print_log(f'search `{entire_k}` in {v}.', logger)
            search_params[entire_k] = v
        if isinstance(v, dict):
            search_params = get_search_params(v, search_params, entire_k,
                                              logger)
    return search_params


def main():

    args = parse_args()

    # do not init the default scope here because it will be init in the runner

    # load config
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    logger = MMLogger.get_instance(name='ParamsSearcher', logger_name='Logger')
    # get all cases
    search_params = get_search_params(cfg.model.tracker, logger=logger)
    search_params_names = tuple(search_params.keys())
    all_search_cases = []
    for values in product(*search_params.values()):
        search = dict()
        for k, v in zip(search_params_names, values):
            search[k] = v
        all_search_cases.append(search)

    print_log(f'Totally {len(all_search_cases)} cases.', logger)

    search_metrics = []
    metrics_types = [cfg.test_evaluator.metric] if isinstance(
        cfg.test_evaluator.metric, str) else cfg.test_evaluator.metric
    if 'HOTA' in metrics_types:
        search_metrics.extend(['HOTA', 'AssA', 'DetA'])
    if 'CLEAR' in metrics_types:
        search_metrics.extend(
            ['MOTA', 'MOTP', 'IDSW', 'TP', 'FN', 'FP', 'Frag', 'MT', 'ML'])
    if 'Identity' in metrics_types:
        search_metrics.extend(['IDF1', 'IDTP', 'IDFN', 'IDFP', 'IDP', 'IDR'])
    print_log(f'Record {search_metrics}.', logger)

    runner = Runner.from_cfg(cfg)
    if is_model_wrapper(runner.model):
        model = runner.model.module
    else:
        model = runner.model

    if args.detector:
        assert not (args.checkpoint and args.detector), \
            'Error: checkpoint and detector checkpoint cannot both exist'
        load_checkpoint(model.detector, args.detector)

    if args.reid:
        assert (args.checkpoint is not None) or (args.detector is not None), \
            'Error: checkpoint and detector checkpoint cannot both not exist'
        assert not (args.checkpoint and args.reid), \
            'Error: checkpoint and reid checkpoint cannot both exist'
        load_checkpoint(model.reid, args.reid)

    for case in all_search_cases:
        for name, value in case.items():
            if hasattr(runner.model, 'module'):
                setattr(runner.model.module.tracker, name, value)
            else:
                setattr(runner.model.tracker, name, value)
        runner.test()
        rank, _ = get_dist_info()
        if rank == 0:
            _records = []
            for metric in search_metrics:
                res = runner.message_hub.get_scalar(
                    'test/motchallenge-metric/' + metric).current()
                if isinstance(res, float):
                    _records.append(f'{res:.3f}')
                else:
                    _records.append(f'{res}')
            print_log(f'-------------- {case}: {_records} --------------',
                      logger)


if __name__ == '__main__':
    main()

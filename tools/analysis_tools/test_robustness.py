# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info
from mmengine.evaluator import DumpResults
from mmengine.fileio import dump
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.registry import RUNNERS
from tools.analysis_tools.robustness_eval import get_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--severities',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5],
        help='corruption severity levels')
    parser.add_argument(
        '--summaries',
        type=bool,
        default=False,
        help='Print summaries for every corruption and severity')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--final-prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default='mPC',
        help='corruption benchmark metric to print at the end')
    parser.add_argument(
        '--final-prints-aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='benchmark',
        help='aggregate all results or only those for benchmark corruptions')
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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.show_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out", "--show" or "show-dir"')

    # load config
    cfg = Config.fromfile(args.config)
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

    cfg.model.backbone.init_cfg.type = None
    cfg.test_dataloader.dataset.test_mode = True

    cfg.load_from = args.checkpoint
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    if 'all' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
            'saturate'
        ]
    elif 'benchmark' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
    elif 'noise' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in args.corruptions:
        corruptions = [
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        ]
    elif 'weather' in args.corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in args.corruptions:
        corruptions = [
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    elif 'holdout' in args.corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in args.corruptions:
        corruptions = ['None']
        args.severities = [0]
    else:
        corruptions = args.corruptions

    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for sev_i, corruption_severity in enumerate(args.severities):
            # evaluate severity 0 (= no corruption) only once
            if corr_i > 0 and corruption_severity == 0:
                aggregated_results[corruption][0] = \
                    aggregated_results[corruptions[0]][0]
                continue

            test_loader_cfg = copy.deepcopy(cfg.test_dataloader)
            # assign corruption and severity
            if corruption_severity > 0:
                corruption_trans = dict(
                    type='Corrupt',
                    corruption=corruption,
                    severity=corruption_severity)
                # TODO: hard coded "1", we assume that the first step is
                # loading images, which needs to be fixed in the future
                test_loader_cfg.dataset.pipeline.insert(1, corruption_trans)

            test_loader = runner.build_dataloader(test_loader_cfg)

            runner.test_loop.dataloader = test_loader
            # set random seeds
            if args.seed is not None:
                runner.set_randomness(args.seed)

            # print info
            print(f'\nTesting {corruption} at severity {corruption_severity}')

            eval_results = runner.test()
            if args.out:
                eval_results_filename = (
                    osp.splitext(args.out)[0] + '_results' +
                    osp.splitext(args.out)[1])
                aggregated_results[corruption][
                    corruption_severity] = eval_results
                dump(aggregated_results, eval_results_filename)

    rank, _ = get_dist_info()
    if rank == 0:
        eval_results_filename = (
            osp.splitext(args.out)[0] + '_results' + osp.splitext(args.out)[1])
        # print final results
        print('\nAggregated results:')
        prints = args.final_prints
        aggregate = args.final_prints_aggregate

        if cfg.dataset_type == 'VOCDataset':
            get_results(
                eval_results_filename,
                dataset='voc',
                prints=prints,
                aggregate=aggregate)
        else:
            get_results(
                eval_results_filename,
                dataset='coco',
                prints=prints,
                aggregate=aggregate)


if __name__ == '__main__':
    main()

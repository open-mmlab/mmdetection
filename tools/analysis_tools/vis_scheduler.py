# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp
import re
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import rich
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn


class SimpleModel(BaseModel):
    """simple model that do nothing in train_step."""

    def __init__(self):
        super().__init__()
        self.data_preprocessor = nn.Identity()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        pass

    def train_step(self, data, optim_wrapper):
        pass


class ParamRecordHook(Hook):

    def __init__(self, by_epoch):
        super().__init__()
        self.by_epoch = by_epoch
        self.lr_list = []
        self.momentum_list = []
        self.wd_list = []
        self.task_id = 0
        self.progress = Progress(BarColumn(), MofNCompleteColumn(),
                                 TextColumn('{task.description}'))

    def before_train(self, runner):
        if self.by_epoch:
            total = runner.train_loop.max_epochs
            self.task_id = self.progress.add_task(
                'epochs', start=True, total=total)
        else:
            total = runner.train_loop.max_iters
            self.task_id = self.progress.add_task(
                'iters', start=True, total=total)
        self.progress.start()

    def after_train_epoch(self, runner):
        if self.by_epoch:
            self.progress.update(self.task_id, advance=1)

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        if not self.by_epoch:
            self.progress.update(self.task_id, advance=1)
        self.lr_list.append(runner.optim_wrapper.get_lr()['lr'][0])
        self.momentum_list.append(
            runner.optim_wrapper.get_momentum()['momentum'][0])
        self.wd_list.append(
            runner.optim_wrapper.param_groups[0]['weight_decay'])

    def after_train(self, runner):
        self.progress.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a Dataset Pipeline')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '-p',
        '--parameter',
        type=str,
        default='lr',
        choices=['lr', 'momentum', 'wd'],
        help='The parameter to visualize its change curve, choose from'
        '"lr", "wd" and "momentum". Defaults to "lr".')
    parser.add_argument(
        '-d',
        '--dataset-size',
        type=int,
        help='The size of the dataset. If specify, `build_dataset` will '
        'be skipped and use this size as the dataset size.')
    parser.add_argument(
        '-n',
        '--ngpus',
        type=int,
        default=1,
        help='The number of GPUs used in training.')
    parser.add_argument(
        '-s',
        '--save-path',
        type=Path,
        help='The learning rate curve plot save path')
    parser.add_argument(
        '--log-level',
        default='WARNING',
        help='The log level of the handler and logger. Defaults to '
        'WARNING.')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--style', type=str, default='whitegrid', help='style of plt')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--window-size',
        default='12*7',
        help='Size of the window to display images, in format of "$W*$H".')
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
    if args.window_size != '':
        assert re.match(r'\d+\*\d+', args.window_size), \
            "'window-size' must be in format 'W*H'."

    return args


def plot_curve(lr_list, args, param_name, iters_per_epoch, by_epoch=True):
    """Plot learning rate vs iter graph."""
    try:
        import seaborn as sns
        sns.set_style(args.style)
    except ImportError:
        pass

    wind_w, wind_h = args.window_size.split('*')
    wind_w, wind_h = int(wind_w), int(wind_h)
    plt.figure(figsize=(wind_w, wind_h))

    ax: plt.Axes = plt.subplot()
    ax.plot(lr_list, linewidth=1)

    if by_epoch:
        ax.xaxis.tick_top()
        ax.set_xlabel('Iters')
        ax.xaxis.set_label_position('top')
        sec_ax = ax.secondary_xaxis(
            'bottom',
            functions=(lambda x: x / iters_per_epoch,
                       lambda y: y * iters_per_epoch))
        sec_ax.set_xlabel('Epochs')
    else:
        plt.xlabel('Iters')
    plt.ylabel(param_name)

    if args.title is None:
        plt.title(f'{osp.basename(args.config)} {param_name} curve')
    else:
        plt.title(args.title)


def simulate_train(data_loader, cfg, by_epoch):
    model = SimpleModel()
    param_record_hook = ParamRecordHook(by_epoch=by_epoch)
    default_hooks = dict(
        param_scheduler=cfg.default_hooks['param_scheduler'],
        runtime_info=None,
        timer=None,
        logger=None,
        checkpoint=None,
        sampler_seed=None,
        param_record=param_record_hook)

    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=data_loader,
        train_cfg=cfg.train_cfg,
        log_level=cfg.log_level,
        optim_wrapper=cfg.optim_wrapper,
        param_scheduler=cfg.param_scheduler,
        default_scope=cfg.default_scope,
        default_hooks=default_hooks,
        visualizer=MagicMock(spec=Visualizer),
        custom_hooks=cfg.get('custom_hooks', None))

    runner.train()

    param_dict = dict(
        lr=param_record_hook.lr_list,
        momentum=param_record_hook.momentum_list,
        wd=param_record_hook.wd_list)

    return param_dict


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.log_level = args.log_level

    # make sure save_root exists
    if args.save_path and not args.save_path.parent.exists():
        raise FileNotFoundError(
            f'The save path is {args.save_path}, and directory '
            f"'{args.save_path.parent}' do not exist.")

    # init logger
    print('Param_scheduler :')
    rich.print_json(json.dumps(cfg.param_scheduler))

    # register all modules in mmdet into the registries
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # prepare data loader
    batch_size = cfg.train_dataloader.batch_size * args.ngpus

    if 'by_epoch' in cfg.train_cfg:
        by_epoch = cfg.train_cfg.get('by_epoch')
    elif 'type' in cfg.train_cfg:
        by_epoch = cfg.train_cfg.get('type') == 'EpochBasedTrainLoop'
    else:
        raise ValueError('please set `train_cfg`.')

    if args.dataset_size is None and by_epoch:
        from mmcls.datasets import build_dataset
        dataset_size = len(build_dataset(cfg.train_dataloader.dataset))
    else:
        dataset_size = args.dataset_size or batch_size

    class FakeDataloader(list):
        dataset = MagicMock(metainfo=None)

    data_loader = FakeDataloader(range(dataset_size // batch_size))
    dataset_info = (
        f'\nDataset infos:'
        f'\n - Dataset size: {dataset_size}'
        f'\n - Batch size per GPU: {cfg.train_dataloader.batch_size}'
        f'\n - Number of GPUs: {args.ngpus}'
        f'\n - Total batch size: {batch_size}')
    if by_epoch:
        dataset_info += f'\n - Iterations per epoch: {len(data_loader)}'
    rich.print(dataset_info + '\n')

    # simulation training process
    param_dict = simulate_train(data_loader, cfg, by_epoch)
    param_list = param_dict[args.parameter]

    if args.parameter == 'lr':
        param_name = 'Learning Rate'
    elif args.parameter == 'momentum':
        param_name = 'Momentum'
    else:
        param_name = 'Weight Decay'
    plot_curve(param_list, args, param_name, len(data_loader), by_epoch)

    if args.save_path:
        plt.savefig(args.save_path)
        print(f'\nThe {param_name} graph is saved at {args.save_path}')

    if not args.not_show:
        plt.show()


if __name__ == '__main__':
    main()

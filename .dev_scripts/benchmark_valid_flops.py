import logging
import re
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmengine import Config, DictAction
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
from mmengine.fileio import FileClient
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner
from modelindex.load_model_index import load
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

console = Console()
MMDET_ROOT = Path(__file__).absolute().parents[1]


def parse_args():
    parser = ArgumentParser(description='Valid all models in model-index.yml')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--checkpoint_root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument('--img', default='demo/demo.jpg', help='Image file')
    parser.add_argument('--models', nargs='+', help='models name to inference')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='The batch size during the inference.')
    parser.add_argument(
        '--flops', action='store_true', help='Get Flops and Params of models')
    parser.add_argument(
        '--flops-str',
        action='store_true',
        help='Output FLOPs and params counts in a string form.')
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
        '--size_divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def inference(config_file, checkpoint, work_dir, args, exp_name):
    logger = MMLogger.get_instance(name='MMLogger')
    logger.warning('if you want test flops, please make sure torch>=1.12')
    cfg = Config.fromfile(config_file)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.log_level = 'WARN'
    cfg.experiment_name = exp_name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # forward the model
    result = {'model': config_file.stem}

    if args.flops:

        if len(args.shape) == 1:
            h = w = args.shape[0]
        elif len(args.shape) == 2:
            h, w = args.shape
        else:
            raise ValueError('invalid input shape')
        divisor = args.size_divisor
        if divisor > 0:
            h = int(np.ceil(h / divisor)) * divisor
            w = int(np.ceil(w / divisor)) * divisor

        input_shape = (3, h, w)
        result['resolution'] = input_shape

        try:
            cfg = Config.fromfile(config_file)
            if hasattr(cfg, 'head_norm_cfg'):
                cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
                cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
                    type='SyncBN', requires_grad=True)
                cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
                    type='SyncBN', requires_grad=True)

            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            model = MODELS.build(cfg.model)
            input = torch.rand(1, *input_shape)
            if torch.cuda.is_available():
                model.cuda()
                input = input.cuda()
            model = revert_sync_batchnorm(model)
            inputs = (input, )
            model.eval()
            outputs = get_model_complexity_info(
                model, input_shape, inputs, show_table=False, show_arch=False)
            flops = outputs['flops']
            params = outputs['params']
            activations = outputs['activations']
            result['Get Types'] = 'direct'
        except:  # noqa 772
            logger = MMLogger.get_instance(name='MMLogger')
            logger.warning(
                'Direct get flops failed, try to get flops with data')
            cfg = Config.fromfile(config_file)
            if hasattr(cfg, 'head_norm_cfg'):
                cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
                cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
                    type='SyncBN', requires_grad=True)
                cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
                    type='SyncBN', requires_grad=True)
            data_loader = Runner.build_dataloader(cfg.val_dataloader)
            data_batch = next(iter(data_loader))
            model = MODELS.build(cfg.model)
            if torch.cuda.is_available():
                model = model.cuda()
            model = revert_sync_batchnorm(model)
            model.eval()
            _forward = model.forward
            data = model.data_preprocessor(data_batch)
            del data_loader
            model.forward = partial(
                _forward, data_samples=data['data_samples'])
            outputs = get_model_complexity_info(
                model,
                input_shape,
                data['inputs'],
                show_table=False,
                show_arch=False)
            flops = outputs['flops']
            params = outputs['params']
            activations = outputs['activations']
            result['Get Types'] = 'dataloader'

        if args.flops_str:
            flops = _format_size(flops)
            params = _format_size(params)
            activations = _format_size(activations)

        result['flops'] = flops
        result['params'] = params

    return result


def show_summary(summary_data, args):
    table = Table(title='Validation Benchmark Regression Summary')
    table.add_column('Model')
    table.add_column('Validation')
    table.add_column('Resolution (c, h, w)')
    if args.flops:
        table.add_column('Flops', justify='right', width=11)
        table.add_column('Params', justify='right')

    for model_name, summary in summary_data.items():
        row = [model_name]
        valid = summary['valid']
        color = 'green' if valid == 'PASS' else 'red'
        row.append(f'[{color}]{valid}[/{color}]')
        if valid == 'PASS':
            row.append(str(summary['resolution']))
            if args.flops:
                row.append(str(summary['flops']))
                row.append(str(summary['params']))
        table.add_row(*row)

    console.print(table)
    table_data = {
        x.header: [Text.from_markup(y).plain for y in x.cells]
        for x in table.columns
    }
    table_pd = pd.DataFrame(table_data)
    table_pd.to_csv('./mmdetection_flops.csv')


# Sample test whether the inference code is correct
def main(args):
    register_all_modules()
    model_index_file = MMDET_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    logger = MMLogger(
        'validation',
        logger_name='validation',
        log_file='benchmark_test_image.log',
        log_level=logging.INFO)

    if args.models:
        patterns = [
            re.compile(pattern.replace('+', '_')) for pattern in args.models
        ]
        filter_models = {}
        for k, v in models.items():
            k = k.replace('+', '_')
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    summary_data = {}
    tmpdir = tempfile.TemporaryDirectory()
    for model_name, model_info in tqdm(models.items()):

        if model_info.config is None:
            continue

        model_info.config = model_info.config.replace('%2B', '+')
        config = Path(model_info.config)

        try:
            config.exists()
        except:  # noqa 722
            logger.error(f'{model_name}: {config} not found.')
            continue

        logger.info(f'Processing: {model_name}')

        http_prefix = 'https://download.openmmlab.com/mmdetection/'
        if args.checkpoint_root is not None:
            root = args.checkpoint_root
            if 's3://' in args.checkpoint_root:
                from petrel_client.common.exception import AccessDeniedError
                file_client = FileClient.infer_client(uri=root)
                checkpoint = file_client.join_path(
                    root, model_info.weights[len(http_prefix):])
                try:
                    exists = file_client.exists(checkpoint)
                except AccessDeniedError:
                    exists = False
            else:
                checkpoint = Path(root) / model_info.weights[len(http_prefix):]
                exists = checkpoint.exists()
            if exists:
                checkpoint = str(checkpoint)
            else:
                print(f'WARNING: {model_name}: {checkpoint} not found.')
                checkpoint = None
        else:
            checkpoint = None

        try:
            # build the model from a config file and a checkpoint file
            result = inference(MMDET_ROOT / config, checkpoint, tmpdir.name,
                               args, model_name)
            result['valid'] = 'PASS'
        except Exception:  # noqa 722
            import traceback
            logger.error(f'"{config}" :\n{traceback.format_exc()}')
            result = {'valid': 'FAIL'}

        summary_data[model_name] = result

    tmpdir.cleanup()
    show_summary(summary_data, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)

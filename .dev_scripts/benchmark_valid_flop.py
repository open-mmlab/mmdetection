import logging
import re
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from time import time

import mmcv
import numpy as np
import torch
from mmengine import Config, DictAction, MMLogger
from mmengine.dataset import Compose, default_collate
from mmengine.fileio import FileClient
from mmengine.runner import Runner
from modelindex.load_model_index import load
from rich.console import Console
from rich.table import Table
import pandas as pd
from rich.text import Text

from mmdet.registry import MODELS
from mmengine import Config
from functools import partial
from mmdet.utils import register_all_modules
from mmengine.runner import Runner
from mmengine.logging import MMLogger
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
from tqdm import tqdm

console = Console()
MMDET_ROOT = Path(__file__).absolute().parents[1]


# classes_map = {
#     'ImageNet-1k': ImageNet.CLASSES,
#     'CIFAR-10': CIFAR10.CLASSES,
#     'CIFAR-100': CIFAR100.CLASSES,
# }

def parse_args():
    parser = ArgumentParser(description='Valid all models in model-index.yml')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--checkpoint-root',
        help='Checkpoint file root path. If set, load checkpoint before test.')
    parser.add_argument('--img', default='demo/demo.jpg', help='Image file')
    parser.add_argument('--models', nargs='+', help='models name to inference')

    parser.add_argument(
        '--inference-time',
        action='store_true',
        help='Test inference time by run 10 times for each model.')
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
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
             'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def inference(config_file, checkpoint, work_dir, args, exp_name):
    cfg = Config.fromfile(config_file)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.log_level = 'WARN'
    cfg.experiment_name = exp_name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the data pipeline
    # test_dataset = cfg.test_dataloader.dataset
    # if test_dataset.pipeline[0]['type'] != 'LoadImageFromFile':
    #     test_dataset.pipeline.insert(0, dict(type='LoadImageFromFile'))
    # if test_dataset.type in ['CIFAR10', 'CIFAR100']:
    #     # The image shape of CIFAR is (32, 32, 3)
    #     test_dataset.pipeline.insert(1, dict(type='Resize', scale=32))

    # from mmdet.structures import DetDataSample
    # from mmengine.structures import InstanceData

    # data_sample = DetDataSample()
    # img_meta = dict(img_shape=(800, 1196, 3),
    #                      pad_shape=(800, 1216, 3))

    # gt_instances = InstanceData(metainfo=img_meta)
    # gt_instances.bboxes = torch.rand((5, 4))
    # gt_instances.labels = torch.rand((5,))
    # data_sample.gt_instances = gt_instances

    # import pdb;pdb.set_trace()
    # data = Compose(test_dataset.pipeline)({'img_path': args.img})
    # data = default_collate([data] * args.batch_size)
    # import pdb;pdb.set_trace()
    # resolution = tuple(data['inputs'].shape[-2:])

    # runner: Runner = Runner.from_cfg(cfg)
    # model = runner.model

    # forward the model
    result = {'model': config_file.stem}

    if args.flops:

        if len(args.shape) == 1:
            h = w = args.shape[0]
        elif len(args.shape) == 2:
            h, w = args.shape
        else:
            raise ValueError('invalid input shape')
        ori_shape = (3, h, w)
        divisor = args.size_divisor
        if divisor > 0:
            h = int(np.ceil(h / divisor)) * divisor
            w = int(np.ceil(w / divisor)) * divisor

        input_shape = (3, h, w)
        result['resolution'] = input_shape

        try:

            cfg = Config.fromfile(config_file)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)

            model = MODELS.build(cfg.model)
            if torch.cuda.is_available():
                model.cuda()
            model.eval()

            flops, activations, params, _, __ = get_model_complexity_info(
                model, input_shape, show_table=False, show_str=False,
                format_size=False)
            result['Get Types'] = 'direct'
        except:
            logger = MMLogger.get_instance(name='MMLogger')
            logger.warning('Direct get flops failed, try to get flops with data')
            cfg = Config.fromfile(config_file)
            data_loader = Runner.build_dataloader(cfg.val_dataloader)
            data_batch = next(iter(data_loader))
            model = MODELS.build(cfg.model)
            _forward = model.forward
            data = model.data_preprocessor(data_batch)
            del data_loader
            model.forward = partial(_forward, data_samples=data['data_samples'])
            flops, activations, params, _, __ = get_model_complexity_info(
                model, input_shape, data['inputs'], show_table=False, show_str=False,
                format_size=False)
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
    # if args.inference_time:
    #     table.add_column('Inference Time (std) (ms/im)')
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
            # if args.inference_time:
            #     time_mean = f"{summary['time_mean']:.2f}"
            #     time_std = f"{summary['time_std']:.2f}"
            #     row.append(f'{time_mean}\t({time_std})'.expandtabs(8))
            if args.flops:
                row.append(str(summary['flops']))
                row.append(str(summary['params']))
        table.add_row(*row)

    console.print(table)
    table_data = {
        x.header: [Text.from_markup(y).plain for y in x.cells] for x in table.columns
    }
    table_pd = pd.DataFrame(table_data)
    table_pd.to_csv("./mmdetection_flops.csv")


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
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
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
        print(model_info.config)
        model_info.config = model_info.config.replace("%2B", "+")
        config = Path(model_info.config)

        try:
            config.exists()
        except:
            logger.error(f'{model_name}: {config} not found.')
            continue

        logger.info(f'Processing: {model_name}')

        http_prefix = 'https://download.openmmlab.com/mmclassification/'
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
        except Exception:
            import traceback
            logger.error(f'"{config}" :\n{traceback.format_exc()}')
            result = {'valid': 'FAIL'}

        summary_data[model_name] = result

    tmpdir.cleanup()
    show_summary(summary_data, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)


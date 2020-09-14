import argparse
import json
import os
import subprocess
import time

import mmcv
import torch
from mmcv import Config, get_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.apis.test import collect_results_gpu
from mmdet.core import encode_mask_results, wrap_fp16_model
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

modelzoo_dict = {
    'configs/atss/atss_r50_fpn_1x_coco.py': {
        'bbox': 39.4
    },
    'configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py': {
        'bbox': 39.3,
        'segm': 35.8
    },
    'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 40.3
    },
    'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 41.2,
        'segm': 35.9
    },
    'configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py': {
        'bbox': 41.2
    },
    'configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py': {
        'bbox': 41.8,
        'ap mask': 37.4
    },
    'configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py': {
        'bbox': 47.4
    },
    'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 40.0
    },
    'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py': {
        'bbox': 38.9
    },
    'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 37.4
    },
    'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 38.2,
        'segm': 34.7
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='The script used for checking the correctness \
            of batch inference')
    parser.add_argument('model_dir', help='directory of models')
    parser.add_argument(
        'json_out', help='the output json records test information like mAP')
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


def check_finish(all_model_dict, result_file):
    tested_cfgs = []
    with open(result_file, 'r+') as f:
        for line in f:
            line = json.loads(line)
            tested_cfgs.append(line['cfg'])
    is_finish = True
    for cfg in sorted(all_model_dict.keys()):
        if cfg not in tested_cfgs:
            is_finish = False
            return cfg
    if is_finish:
        with open(result_file, 'a+') as f:
            f.write('finished\n')


def dump_dict(record_dict, json_out):
    with open(json_out, 'a+') as f:
        mmcv.dump(record_dict, f, file_format='json')
        f.write('\n')


def main():
    args = parse_args()
    # init distributed env first, since logger depends on the dist
    # info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, backend='nccl')
    rank, world_size = get_dist_info()

    logger = get_logger('root')

    # read info of checkpoints and config
    result_dict = dict()
    for model_family_dir in os.listdir(args.model_dir):
        model_family_dir_path = os.path.join(args.model_dir, model_family_dir)
        for model_dir in os.listdir(model_family_dir_path):
            model_dir_path = os.path.join(model_family_dir_path, model_dir)
            cpt = [f for f in os.listdir(model_dir_path) if f.endswith('.pth')]
            assert len(cpt) == 1, f'no valid checkpoint for {model_dir}'
            cpt = cpt[0]
            cpt_path = os.path.join(model_dir_path, cpt)
            config_path = os.path.join('configs', model_family_dir,
                                       model_dir) + '.py'
            # check if the config name in modelzoo dict
            assert config_path in modelzoo_dict, \
                f'{config_path} not in modelzoo dict'
            assert os.path.isfile(
                config_path), f'no valid config path for {config_path}'
            result_dict[config_path] = cpt_path

    cfg = check_finish(result_dict, args.json_out)
    cpt = result_dict[cfg]
    try:
        cfg_name = cfg
        logger.info(f'evaluate {cfg}')
        record = dict(cfg=cfg, cpt=cpt)
        cfg = Config.fromfile(cfg)
        # cfg.data.test.ann_file = 'data/val_0_10.json'
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True

        # build the dataloader
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, cpt, map_location='cpu')
        # old versions did not save class info in checkpoints,
        # this walkaround is for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            model.eval()
            results = []
            dataset = data_loader.dataset
            rank, world_size = get_dist_info()
            if rank == 0:
                prog_bar = mmcv.ProgressBar(len(dataset))
            # This line can prevent deadlock problem in some cases.
            time.sleep(2)
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    if 'faster_rcnn' in cpt and i == 20 and rank == 0:
                        record[3333333]
                    result = model(return_loss=False, rescale=True, **data)
                    # encode mask results
                    if isinstance(result[0], tuple):
                        result = [(bbox_results,
                                   encode_mask_results(mask_results))
                                  for bbox_results, mask_results in result]
                results.extend(result)

                if rank == 0:
                    batch_size = len(result)
                    for _ in range(batch_size * world_size):
                        prog_bar.update()

            # collect results from all ranks
            outputs = collect_results_gpu(results, len(dataset))

        if rank == 0:
            ref_mAP_dict = modelzoo_dict[cfg_name]
            metrics = list(ref_mAP_dict.keys())
            eval_results = dataset.evaluate(outputs, metrics)
            print(eval_results)
            for metric in metrics:
                eval_metric = eval_results[f'{metric}_mAP']
                record[metric] = eval_metric
            dump_dict(record, args.json_out)
            check_finish(result_dict, args.json_out)
    except Exception as e:
        logger.error(f'rank: {rank} test fail with error: {e}')
        record['terminate'] = True
        dump_dict(record, args.json_out)
        check_finish(result_dict, args.json_out)
        subprocess.call('xxx')


if __name__ == '__main__':
    main()

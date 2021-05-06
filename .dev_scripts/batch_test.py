"""
some instructions
1. Fill the models that needs to be checked in the modelzoo_dict
2. Arange the structure of the directory as follows, the script will find the
   corresponding config itself:
   model_dir/model_family/checkpoints
   e.g.: models/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
         models/faster_rcnn/faster_rcnn_r101_fpn_1x_coco_20200130-047c8118.pth
3. Excute the batch_test.sh
"""

import argparse
import json
import os
import subprocess

import mmcv
import torch
from mmcv import Config, get_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

modelzoo_dict = {
    'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 0.374
    },
    'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py': {
        'bbox': 0.382,
        'segm': 0.347
    },
    'configs/rpn/rpn_r50_fpn_1x_coco.py': {
        'AR@1000': 0.582
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
    # check if all models are checked
    tested_cfgs = []
    with open(result_file, 'r+') as f:
        for line in f:
            line = json.loads(line)
            tested_cfgs.append(line['cfg'])
    is_finish = True
    for cfg in sorted(all_model_dict.keys()):
        if cfg not in tested_cfgs:
            return cfg
    if is_finish:
        with open(result_file, 'a+') as f:
            f.write('finished\n')


def dump_dict(record_dict, json_out):
    # dump result json dict
    with open(json_out, 'a+') as f:
        mmcv.dump(record_dict, f, file_format='json')
        f.write('\n')


def main():
    args = parse_args()
    # touch the output json if not exist
    with open(args.json_out, 'a+'):
        pass
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
        for model in os.listdir(
                os.path.join(args.model_dir, model_family_dir)):
            # cpt: rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth
            # cfg: rpn_r50_fpn_1x_coco.py
            cfg = model.split('.')[0][:-18] + '.py'
            cfg_path = os.path.join('configs', model_family_dir, cfg)
            assert os.path.isfile(
                cfg_path), f'{cfg_path} is not valid config path'
            cpt_path = os.path.join(args.model_dir, model_family_dir, model)
            result_dict[cfg_path] = cpt_path
            assert cfg_path in modelzoo_dict, f'please fill the ' \
                                              f'performance of cfg: {cfg_path}'
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
        samples_per_gpu = 2  # hack test with 2 image per gpu
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
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, cpt, map_location='cpu')
        # old versions did not save class info in checkpoints,
        # this walkaround is for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
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
            outputs = multi_gpu_test(model, data_loader, 'tmp')
        if rank == 0:
            ref_mAP_dict = modelzoo_dict[cfg_name]
            metrics = list(ref_mAP_dict.keys())
            metrics = [
                m if m != 'AR@1000' else 'proposal_fast' for m in metrics
            ]
            eval_results = dataset.evaluate(outputs, metrics)
            print(eval_results)
            for metric in metrics:
                if metric == 'proposal_fast':
                    ref_metric = modelzoo_dict[cfg_name]['AR@1000']
                    eval_metric = eval_results['AR@1000']
                else:
                    ref_metric = modelzoo_dict[cfg_name][metric]
                    eval_metric = eval_results[f'{metric}_mAP']
                if abs(ref_metric - eval_metric) > 0.003:
                    record['is_normal'] = False
            dump_dict(record, args.json_out)
            check_finish(result_dict, args.json_out)
    except Exception as e:
        logger.error(f'rank: {rank} test fail with error: {e}')
        record['terminate'] = True
        dump_dict(record, args.json_out)
        check_finish(result_dict, args.json_out)
        # hack there to throw some error to prevent hang out
        subprocess.call('xxx')


if __name__ == '__main__':
    main()

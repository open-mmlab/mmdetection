import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
sys.path.append('/mnt/lustre/pangjiangmiao/sensenet_folder/mmcv')
import argparse

import numpy as np
import torch

import mmcv
from mmcv import Config
from mmcv.torchpack import load_checkpoint, parallel_test
from mmdet.core import _data_func, results2json
from mmdet.datasets import CocoDataset
from mmdet.datasets.data_engine import build_data
from mmdet.models import Detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--out_json', action='store_true', help='get json output file')
    args = parser.parse_args()
    return args


args = parse_args()


def main():
    cfg = Config.fromfile(args.config)
    cfg.model['pretrained'] = None
    # TODO this img_per_gpu
    cfg.img_per_gpu == 1

    if args.world_size == 1:
        # TODO verify this part
        args.dist = False
        args.img_per_gpu = cfg.img_per_gpu
        args.data_workers = cfg.data_workers
        model = Detector(**cfg.model, **meta_params)
        load_checkpoint(model, args.checkpoint)
        test_loader = build_data(cfg.test_dataset, args)
        model = torch.nn.DataParallel(model, device_ids=0)
        # TODO write single_test
        outputs = single_test(test_loader, model)
    else:
        test_dataset = CocoDataset(**cfg.test_dataset)
        model = dict(cfg.model, **cfg.meta_params)
        outputs = parallel_test(Detector, model,
                                args.checkpoint, test_dataset, _data_func,
                                range(args.world_size))

    if args.out:
        mmcv.dump(outputs, args.out, protocol=4)
        if args.out_json:
            results2json(test_dataset, outputs, args.out + '.json')


if __name__ == '__main__':
    main()

import argparse

import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmdet.models import build_detector, detectors


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output', help='onnx file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    assert getattr(detectors, cfg.model['type']) is detectors.SingleStageDetector
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])

    batch = torch.FloatTensor(1, 3, cfg.input_size, cfg.input_size).cuda()
    data = dict(img=batch, img_meta=[{'img_shape': (cfg.input_size, cfg.input_size, 3),
                                      'scale_factor': np.array([1, 1, 1, 1], dtype=np.float32)}])
    model.eval()
    model.module.onnx_export(export_name=args.output, **data)


if __name__ == '__main__':
    main()

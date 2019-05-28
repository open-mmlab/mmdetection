import argparse

import numpy as np
import torch
from mmcv.parallel import MMDataParallel

from mmdet.apis import init_detector
from mmdet.models import detectors


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet onnx exporter for \
                                                  SSD detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output', help='onnx file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint)
    cfg = model.cfg
    assert getattr(detectors, cfg.model['type']) is \
        detectors.SingleStageDetector
    model = MMDataParallel(model, device_ids=[0])

    batch = torch.FloatTensor(1, 3, cfg.input_size, cfg.input_size).cuda()
    input_shape = (cfg.input_size, cfg.input_size, 3)
    scale = np.array([1, 1, 1, 1], dtype=np.float32)
    data = dict(img=batch, img_meta=[{'img_shape': input_shape,
                                      'scale_factor': scale}])
    model.eval()
    model.module.onnx_export(export_name=args.output, **data)


if __name__ == '__main__':
    main()

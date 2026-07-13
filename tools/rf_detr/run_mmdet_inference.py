from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope
from PIL import Image

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
import mmdet.models  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    DefaultScope.get_instance('rf-detr-mmdet-infer', scope_name='mmdet')
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
    model.to(args.device).eval()

    image = Image.open(args.image).convert('RGB')
    arr = np.asarray(image).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0).to(args.device)
    data_sample = DetDataSample()
    data_sample.set_metainfo(
        dict(
            img_shape=(image.height, image.width),
            ori_shape=(image.height, image.width),
            batch_input_shape=(image.height, image.width)))
    with torch.no_grad():
        pred = model(tensor, [data_sample], mode='predict')[0].pred_instances
    np.savez(
        args.output,
        bboxes=pred.bboxes.cpu().numpy(),
        scores=pred.scores.cpu().numpy(),
        labels=pred.labels.cpu().numpy())
    print(json.dumps({'detections': len(pred.scores)}))


if __name__ == '__main__':
    main()

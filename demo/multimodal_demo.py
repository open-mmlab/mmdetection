# Copyright (c) OpenMMLab. All rights reserved.
"""MultiModal Demo.

Example:
    python demo/multimodal_demo.py demo/demo.jpg bench \
    configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
    https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

    python demo/multimodal_demo.py demo/demo.jpg "bench . car . " \
    configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
    https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

    python demo/multimodal_demo.py demo/demo.jpg "bench . car . "  -c \
    configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
    https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

    python demo/multimodal_demo.py demo/demo.jpg \
    "There are a lot of cars here." \
    configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
    https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth
"""

import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmengine.utils import path

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image path, include image file and URL.')
    parser.add_argument('text', help='text prompt')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    result = inference_detector(
        model,
        args.img,
        text_prompt=args.text,
        custom_entities=args.custom_entities)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    out_file = None
    if not args.show:
        path.mkdir_or_exist(args.out_dir)
        out_file = osp.join(args.out_dir, osp.basename(args.img))

    visualizer.add_datasample(
        'results',
        img,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=0,
        out_file=out_file,
        pred_score_thr=args.score_thr)

    if out_file:
        print(f'\nResults have been saved at {osp.abspath(out_file)}')


if __name__ == '__main__':
    main()

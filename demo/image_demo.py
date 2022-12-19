# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import warnings
from argparse import ArgumentParser

import mmcv
from mmengine.logging import print_log

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector)
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--out-dir', default='outputs', help='Dir to output file')
    parser.add_argument(
        '--no-save', action='store_true', help='Do not save detection results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    if args.no_save and not args.show:
        warnings.warn('It doesn\'t make sense to neither save the prediction '
                      'result nor display it. Force set args.no_save to False')
        args.no_save = False

    if not os.path.exists(args.out_dir) and not args.no_save:
        os.mkdir(args.out_dir)

    # TODO: Support inference of image directory.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    filename = os.path.basename(args.img)
    if args.no_save:
        out_file = None
    else:
        out_file = os.path.join(args.out_dir, filename)

    visualizer.add_datasample(
        filename,
        img,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=0,
        out_file=out_file,
        pred_score_thr=args.score_thr)

    if out_file is not None:
        print_log(f'\nResults have been saved at {out_file}')


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        pred_sample=result[0],
        show=args.out_file is None,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    assert not args.async_test, 'async inference is not supported yet.'
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)

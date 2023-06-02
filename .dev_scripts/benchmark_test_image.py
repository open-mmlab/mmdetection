# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path')
    parser.add_argument('--img', default='demo/demo.jpg', help='Image file')
    parser.add_argument('--aug', action='store_true', help='aug test')
    parser.add_argument('--model-name', help='model name to inference')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--out-dir', default=None, help='Dir to output file')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def inference_model(config_name, checkpoint, visualizer, args, logger=None):
    cfg = Config.fromfile(config_name)
    if args.aug:
        raise NotImplementedError()

    model = init_detector(
        cfg, checkpoint, palette=args.palette, device=args.device)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    if args.show or args.out_dir is not None:
        img = mmcv.imread(args.img)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        out_file = None
        if args.out_dir is not None:
            out_dir = args.out_dir
            mkdir_or_exist(out_dir)

            out_file = osp.join(
                out_dir,
                config_name.split('/')[-1].replace('py', 'jpg'))

        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=args.wait_time,
            out_file=out_file,
            pred_score_thr=args.score_thr)

    return result


# Sample test whether the inference code is correct
def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    config = Config.fromfile(args.config)

    # init visualizer
    visualizer_cfg = dict(type='DetLocalVisualizer', name='visualizer')
    visualizer = VISUALIZERS.build(visualizer_cfg)

    # test single model
    if args.model_name:
        if args.model_name in config:
            model_infos = config[args.model_name]
            if not isinstance(model_infos, list):
                model_infos = [model_infos]
            model_info = model_infos[0]
            config_name = model_info['config'].strip()
            print(f'processing: {config_name}', flush=True)
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info['checkpoint'].strip())
            # build the model from a config file and a checkpoint file
            inference_model(config_name, checkpoint, visualizer, args)
            return
        else:
            raise RuntimeError('model name input error.')

    # test all model
    logger = MMLogger.get_instance(
        name='MMLogger',
        log_file='benchmark_test_image.log',
        log_level=logging.ERROR)

    for model_key in config:
        model_infos = config[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'], flush=True)
            config_name = model_info['config'].strip()
            checkpoint = osp.join(args.checkpoint_root,
                                  model_info['checkpoint'].strip())
            try:
                # build the model from a config file and a checkpoint file
                inference_model(config_name, checkpoint, visualizer, args,
                                logger)
            except Exception as e:
                logger.error(f'{config_name} " : {repr(e)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

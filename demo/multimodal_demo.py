# Copyright (c) OpenMMLab. All rights reserved.

from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    from mmengine.runner import save_checkpoint
    save_checkpoint(dict(state_dict=model.state_dict()), 'mmdet.pth')

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    result = inference_detector(model, args.img, text_prompt=args.text)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    result.pred_instances = result.pred_instances[
        result.pred_instances.scores > args.score_thr]
    print(result.pred_instances)

    visualizer.add_datasample(
        'xxx',
        img,
        data_sample=result,
        draw_gt=False,
        show=True,
        wait_time=0,
        out_file=None,
        pred_score_thr=0)


if __name__ == '__main__':
    main()

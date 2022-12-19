# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.logging import print_log
from mmengine.utils import track_iter_progress

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--out-dir', default='outputs', help='Dir to output file')
    parser.add_argument(
        '--no-save', action='store_true', help='Do not save detection results')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=1,
        help='the interval of show (s)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.no_save and not args.show:
        warnings.warn('It doesn\'t make sense to neither save the prediction '
                      'result nor display it. Force set args.no_save to False')
        args.no_save = False

    if not os.path.exists(args.out_dir) and not args.no_save:
        os.mkdir(args.out_dir)

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None

    filename = os.path.basename(args.video)
    if args.no_save:
        out_file = None
    else:
        out_file = os.path.join(args.out_dir, filename)

    if out_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            out_file, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in track_iter_progress(video_reader):
        result = inference_detector(model, frame, test_pipeline=test_pipeline)
        visualizer.add_datasample(
            name='video',
            image=frame,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=args.score_thr)
        frame = visualizer.get_image()

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.show_interval)
        if out_file is not None:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    if out_file is not None:
        print_log(f'\nResults have been saved at {out_file}')


if __name__ == '__main__':
    main()

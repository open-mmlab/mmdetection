# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine

from mmdet.apis import inference_mot, init_track_model
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


def main(args):
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    # define output
    OUT_VIDEO = False
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    register_all_modules(init_default_scope=True)

    # build the model from a config file and a checkpoint file
    model = init_track_model(args.config, args.checkpoint, device=args.device)

    # build the visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.input, img)
            img = mmcv.imread(img_path)
        # result [TrackDataSample]
        result = inference_mot(model, img, frame_id=i, video_len=len(imgs))
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        # show the results
        visualizer.add_datasample(
            'mot',
            img[..., ::-1],
            data_sample=result,
            show=args.show,
            draw_gt=False,
            out_file=out_file,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=args.score_thr,
            step=i)

        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    args = parse_args()
    main(args)

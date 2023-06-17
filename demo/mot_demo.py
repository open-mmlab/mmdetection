# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
from mmengine.registry import init_default_scope

from mmdet.apis import inference_mot, init_track_model
from mmdet.registry import VISUALIZERS

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--detector', help='det checkpoint file')
    parser.add_argument('--reid', help='reid checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--out', help='output video file (mp4 format) or folder')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


def main(args):
    assert args.out or args.show
    # load images
    if osp.isdir(args.inputs):
        imgs = sorted(
            filter(lambda x: x.endswith(IMG_EXTENSIONS),
                   os.listdir(args.inputs)),
            key=lambda x: int(x.split('.')[0]))
        in_video = False
    else:
        imgs = mmcv.VideoReader(args.inputs)
        in_video = True

    # define output
    out_video = False
    if args.out is not None:
        if args.out.endswith('.mp4'):
            out_video = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.out.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.out
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or out_video:
        if fps is None and in_video:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    init_default_scope('mmdet')

    # build the model from a config file and a checkpoint file
    model = init_track_model(
        args.config,
        args.checkpoint,
        args.detector,
        args.reid,
        device=args.device)

    # build the visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    prog_bar = mmengine.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img_path = osp.join(args.inputs, img)
            img = mmcv.imread(img_path)
        # result [TrackDataSample]
        result = inference_mot(model, img, frame_id=i, video_len=len(imgs))
        if args.out is not None:
            if in_video or out_video:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None

        # show the results
        visualizer.add_datasample(
            'mot',
            img[..., ::-1],
            data_sample=result[0],
            show=args.show,
            draw_gt=False,
            out_file=out_file,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=args.score_thr,
            step=i)

        prog_bar.update()

    if args.out and out_video:
        print(f'making the output video at {args.out} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.out, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    args = parse_args()
    main(args)

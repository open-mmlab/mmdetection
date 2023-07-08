# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from typing import Tuple

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress

from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample

try:
    import ffmpegcv
except ImportError:
    raise ImportError(
        'Please install ffmpegcv with:\n\n    pip install ffmpegcv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDetection video demo with GPU acceleration')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--nvdecode', action='store_true', help='Use NVIDIA decoder')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def prefetch_batch_input_shape(model: nn.Module, ori_wh: Tuple[int,
                                                               int]) -> dict:
    cfg = model.cfg
    w, h = ori_wh
    cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = {'img': np.zeros((h, w, 3), dtype=np.uint8), 'img_id': 0}
    data = test_pipeline(data)
    data['inputs'] = [data['inputs']]
    data['data_samples'] = [data['data_samples']]
    data_sample = model.data_preprocessor(data, False)['data_samples']
    batch_input_shape = data_sample[0].batch_input_shape
    return batch_input_shape


def pack_data(frame_resize: np.ndarray, batch_input_shape: Tuple[int, int],
              ori_shape: Tuple[int, int]) -> dict:
    assert frame_resize.shape[:2] == batch_input_shape
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape':
        batch_input_shape,
        'ori_shape':
        ori_shape,
        'scale_factor': (batch_input_shape[0] / ori_shape[0],
                         batch_input_shape[1] / ori_shape[1])
    })
    frame_resize = torch.from_numpy(frame_resize).permute((2, 0, 1)).cuda()
    data = {'inputs': [frame_resize], 'data_samples': [data_sample]}
    return data


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    if args.nvdecode:
        VideoCapture = ffmpegcv.VideoCaptureNV
    else:
        VideoCapture = ffmpegcv.VideoCapture
    video_origin = VideoCapture(args.video)

    batch_input_shape = prefetch_batch_input_shape(
        model, (video_origin.width, video_origin.height))
    ori_shape = (video_origin.height, video_origin.width)
    resize_wh = batch_input_shape[::-1]
    video_resize = VideoCapture(
        args.video,
        resize=resize_wh,
        resize_keepratio=True,
        resize_keepratioalign='topleft')

    video_writer = None
    if args.out:
        video_writer = ffmpegcv.VideoWriter(args.out, fps=video_origin.fps)

    with torch.no_grad():
        for i, (frame_resize, frame_origin) in enumerate(
                zip(track_iter_progress(video_resize), video_origin)):
            data = pack_data(frame_resize, batch_input_shape, ori_shape)
            result = model.test_step(data)[0]

            visualizer.add_datasample(
                name='video',
                image=frame_origin,
                data_sample=result,
                draw_gt=False,
                show=False,
                pred_score_thr=args.score_thr)

            frame_mask = visualizer.get_image()

            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame_mask, 'video', args.wait_time)
            if args.out:
                video_writer.write(frame_mask)

    if video_writer:
        video_writer.release()
    video_origin.release()
    video_resize.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

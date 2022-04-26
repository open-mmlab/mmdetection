import argparse

import cv2
import mmcv
import ffmpegcv
import torch
import tqdm
import numpy as np
from torchvision.transforms import functional as F
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument('--nvdecode', action='store_true', help='Use NVIDIA decorder')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def prepare_cfg(model, ori_wh):
    cfg = model.cfg
    w, h = ori_wh
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'img':np.zeros((h, w, 3), dtype=np.uint8)}
    data = test_pipeline(data)
    data_template = data['img_metas'][0].data
    mean = torch.tensor(data_template['img_norm_cfg']['mean'])
    std = torch.tensor(data_template['img_norm_cfg']['std'])
    return data_template, mean, std


def prepare_img(data_template, mean, std, frame):
    assert frame.shape==data_template['pad_shape']
    frame_cuda = torch.from_numpy(frame.astype(np.float32)).cuda()
    frame_cuda = frame_cuda.permute(2, 0, 1) # HWC to CHW
    frame_cuda = F.normalize(
            frame_cuda, mean=mean, std=std, inplace=True)
    frame_cuda = frame_cuda[None, :, :, :]    # NCHW
    data = {'img': [frame_cuda], 'img_metas': [[data_template]]}
    return data


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    VideoCapture = ffmpegcv.VideoCaptureNV if args.nvdecode else ffmpegcv.VideoCapture
    video_origin = VideoCapture(args.video)
    data_template, mean, std = prepare_cfg(model, 
                                           (video_origin.width, video_origin.height))
    resize_wh = data_template['pad_shape'][1::-1]
    video_reader = VideoCapture(args.video,
                                resize = resize_wh,
                                resize_keepratio = True,
                                resize_keepratioalign = 'topleft',
                                pix_fmt='rgb24')
    video_writer = None
    if args.out:
        video_writer = ffmpegcv.VideoWriter(args.out, fps=video_reader.fps)

    with torch.cuda.device(args.device), torch.no_grad():
        for frame, frame_origin in zip(tqdm.tqdm(video_reader), video_origin):
            data = prepare_img(data_template, mean, std, frame)
            result = model(return_loss=False, rescale=True, **data)[0]
            frame_mask = model.show_result(frame_origin, result, score_thr=args.score_thr)
            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame_mask, 'video', args.wait_time)
            if args.out:
                video_writer.write(frame_mask)

    if video_writer:
        video_writer.release()
    video_origin.release()
    video_reader.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import argparse

import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector


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
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_capture = cv2.VideoCapture(args.video)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out, fourcc, fps,
                                       (frame_width, frame_height))
    else:
        video_writer = None
    prog_bar = mmcv.ProgressBar(num_frames)
    ind = 0
    while ind < num_frames:
        ind += 1
        prog_bar.update()
        ret, frame = video_capture.read()
        if ret:
            result = inference_detector(model, frame)
            frame = model.show_result(frame, result, score_thr=args.score_thr)
            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', args.wait_time)
            if video_writer:
                video_writer.write(frame)
        else:
            print(f'Fail to read {args.video} video')
            break

    video_capture.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

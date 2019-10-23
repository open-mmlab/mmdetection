import argparse

import cv2
import torch
import json

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    camera = cv2.VideoCapture(args.camera_id)
    # writer = cv2.VideoWriter('/tmp/video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))
    raw_out = []

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        if not ret_val:
            break
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        raw_out.append(result)

        xxx = show_result(
            img, result, model.CLASSES, score_thr=args.score_thr, show=True, wait_time=1)
        # print(xxx)
        # writer.write(xxx)

    # writer.release()
    # with open(args.raw_out, 'w') as f:
    #     json.dump(raw_out, f)

if __name__ == '__main__':
    main()

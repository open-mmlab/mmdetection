from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import cv2
import numpy as np

config_file = '../configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py'
checkpoint_file = '../checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

cap = cv2.VideoCapture('test.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
target_fps   = round(cap.get(cv2.CAP_PROP_FPS))
frame_width  = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames   = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

video_writer = cv2.VideoWriter('output.mp4', fourcc, target_fps, (frame_width, frame_height))
counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        result = inference_detector(model, frame)
        result = model.show_result(frame, result, score_thr=0.3, show=False)
        print(f'prorocess frame {counter}/{num_frames}...')
        counter += 1
    else:
        break
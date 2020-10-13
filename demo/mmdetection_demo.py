from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import time
import sys
import glob
import os
from datetime import datetime

def process_video_crcnn(frame_offset, frame_count, config_file, checkpoint_file, video_path):
    """
    frame_offset: skipping this many frames
    frame_count:  run detection on this many frames
    """

    f_number = 0
    frame_offset = int(frame_offset)
    frame_count = int(frame_count)

    video = mmcv.VideoReader(video_path)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.cfg.data.test.pipeline[1]['img_scale'] = video.resolution
    
    print('[config] img_scale: {}'.format(model.cfg.data.test.pipeline[1]['img_scale']))
    print('[config] score threshold: {}'.format(model.cfg.test_cfg['rcnn']['score_thr']))
    print('[config] iou threshold: {}'.format(model.cfg.test_cfg['rcnn']['nms']['iou_threshold']))
    print('[config] rpn nms threshold: {}'.format(model.cfg.test_cfg['rpn']['nms_thr']))

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")

    log_filename = './demo/dump/det.txt'
    log_file = open(log_filename, 'w')

    start_process = time.time()
    
    slice_start = 0 if frame_offset == 0 else frame_offset-1
    slice_end = frame_offset+frame_count

    print('[DBG] processing frames from {} - {}'.format(range(slice_start,slice_end)[0], range(slice_start,slice_end)[-1]))

    last_boxes = []

    for index in range(slice_start,slice_end):
        frame = video[index]
        f_number = f_number + 1
        
        if frame is None:
            print('[DBG] Empty frame received!')
            break

        start_time = time.time()
        result = inference_detector(model, frame)
        end_time = time.time()
        
        bbox_result, _ = result, None
        bboxes = np.vstack(bbox_result)

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        if len(bboxes) == 0 or (len(bboxes) == 1 and labels[0] != 1):
            if len(last_boxes) == 0:
                print('[DBG] both current & previous detection lists for frame %d are empty' % (f_number))
                log_file.write(str(f_number)+","+str(100.0)+","+str(100.0)+","+str(135.0)+","+str(228.0)+","+str(0.1) + "\n")
            else:
                print('[DBG] received empty detection list for frame %d copying boxes from previous frame' % (f_number))
                for i in range(len(last_boxes)):
                    box = last_boxes[i]
                    d = (box[0], box[1], box[2], box[3], box[4])
                    # cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255,0,0), 2)
                    log_file.write(str(f_number)+","+str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3])+","+str(d[4]) + "\n")
        else:
            for i in range(len(bboxes)):
                # bb [816.4531     265.64264    832.7383     311.08356      0.99859136]
                bb = bboxes[i]
                if labels[i] != 1:
                    continue

                d = (bb[0], bb[1], bb[2], bb[3], bb[4])
                cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255,0,0), 2)
                log_file.write(str(f_number)+","+str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3])+","+str(d[4]) + "\n")

            last_boxes = bboxes.copy()

        if f_number == 1 or f_number % 300 == 0:
            end_process = time.time()
            print('[DBG][{}/{}] frame inference time: {} {}, elapsed time: {} {}'.format(f_number+slice_start, slice_end-1, end_time-start_time, '.s', (end_process-start_process), '.s'))

        if f_number == 1 or f_number % 3000 == 0:
            dump_path = "./demo/dump/dump-%06d.jpg" % (f_number)
            cv2.imwrite(dump_path, frame)
            log_file.flush()
            os.fsync(log_file.fileno())

    print('[DBG] detection complete!')
    log_file.close()

def process_jpg_crcnn(config_file, checkpoint_file, image_dir):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")

    log_filename = './demo/dump/det.txt'
    log_file = open(log_filename, 'w')

    start_process = time.time()

    #dsort_img_path = '/home/dmitriy.khvan/dsort-gcp/bepro-data/data/img1'
    frame_count = len(glob.glob(os.path.join(image_dir,'*.jpg')))
    
    for num, filename in enumerate(sorted(glob.glob(os.path.join(image_dir,'*.jpg')))):
        f_number = num + 1
        frame = cv2.imread(filename)

        if frame is None:
            break
    
        start_time = time.time()
        result = inference_detector(model, frame)
        end_time = time.time()
        
        bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        for i in range(len(bboxes)):
            bb = bboxes[i]
            if labels[i] != 0:  continue
            d = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255,0,0), 2)
            log_file.write(str(f_number)+","+str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3]) + "\n")

        if f_number == 1 or f_number % 500 == 0:
            end_process = time.time()
            print('[DBG][{}/{}] frame inference time: {} {}, elapsed time: {} {}'.format(f_number, frame_count, end_time-start_time, '.s', (end_process-start_process), '.s'))
            
        if f_number == 1 or f_number % 1000 == 0:
            dump_path = "./demo/dump/dump-%06d.jpg" % (f_number)
            cv2.imwrite(dump_path, frame)
            log_file.flush()
            os.fsync(log_file.fileno())

    print('[DBG] detection complete!')
    log_file.close()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    config_file = sys.argv[2]
    checkpoint_file = sys.argv[3]
    frame_offset = sys.argv[4]
    frame_count = sys.argv[5]

    # python demo/mmdetection_demo.py PVO4R8Dh-trim.mp4 configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_bepro.py checkpoint/crcnn_r50_bepro_stitch.pth 0 87150 /home/dmitriy.khvan/tmp/   
    process_video_crcnn(frame_offset, frame_count, config_file, checkpoint_file, data_dir)
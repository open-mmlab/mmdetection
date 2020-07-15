from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import time
import sys
import glob
import os
from datetime import datetime

def process_video_crcnn(frame_offset, frame_count, config_file, checkpoint_file, video_path, dsort_img_path):
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
    print('[config] iou threshold: {}'.format(model.cfg.test_cfg['rcnn']['nms']['iou_thr']))
    print('[config] rpn nms threshold: {}'.format(model.cfg.test_cfg['rpn']['nms_thr']))

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")

    log_filename = './demo/dump/det.txt'
    log_file = open(log_filename, 'w')

    start_process = time.time()

    slice_start = frame_offset
    slice_end = frame_offset+frame_count

    print('[DBG] processing frames from {} - {}'.format(range(slice_start,slice_end)[0], range(slice_start,slice_end)[-1]))

    last_boxes = []

    for index in range(slice_start,slice_end):
        # debug: frame slices
        if index == slice_start or index == (slice_end-1):
            if index == slice_start:
                dump_frame_path = './demo/dump/_start_frame.jpg'
            else:
                dump_frame_path = './demo/dump/_end_frame.jpg'
            cv2.imwrite(dump_frame_path, video[index])

        frame = video[index]
        f_number = f_number + 1
        
        if frame is None:
            print('[DBG] Empty frame received!')
            break

        # dump frame for tracking
        f_id = "%05d.jpg" % f_number
        img_dsort = os.path.join(dsort_img_path,f_id)
        cv2.imwrite(img_dsort, frame)
    
        start_time = time.time()
        result = inference_detector(model, frame)
        end_time = time.time()
        
        bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        if len(bboxes) < 1:
            if len(last_boxes) < 1:
                log_file.write(str(f_number)+","+str(100)+","+str(100)+","+str(135)+","+str(228) + "\n")
            else:
                for i in range(len(last_boxes)):
                    box = last_boxes[i]
                    d = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255,0,0), 2)
                    log_file.write(str(f_number)+","+str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3]) + "\n")
        else:
            for i in range(len(bboxes)):
                bb = bboxes[i]
                if labels[i] != 1:  
                    continue
                d = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255,0,0), 2)
                log_file.write(str(f_number)+","+str(d[0])+","+str(d[1])+","+str(d[2])+","+str(d[3]) + "\n")

            last_boxes = bboxes.copy()

        if f_number == 1 or f_number % 300 == 0:
            end_process = time.time()
            print('[DBG][{}/{}] frame inference time: {} {}, elapsed time: {} {}'.format(f_number+slice_start, slice_end-1, end_time-start_time, '.s', (end_process-start_process), '.s'))

        if f_number == 1 or f_number % 1000 == 0:
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
    dsort_dir = sys.argv[6]

    process_video_crcnn(frame_offset, frame_count, config_file, checkpoint_file, data_dir, dsort_dir)
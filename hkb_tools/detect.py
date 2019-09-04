import argparse

import os.path as osp
import os
import cv2
import sys
import glob
import json
import time
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.datasets import HKBDataset
from mmdet.models import build_detector, detectors
from mmdet.apis import init_detector, inference_detector, show_result
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--dataset', type=str, default='HKB',
                        choices=['VisDrone', 'HKB'], help='dataset name')
    # parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--config', default="hkb_tools/cascade_rcnn_x101_32x4d_fpn_1x.py", help='test config file path')
    parser.add_argument('--checkpoint', default="hkb_tools/work_dirs/epoch_32.pth", help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def show_result(img, result, classes, score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = classes
    bboxes = np.vstack(result)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    img1 = img.copy()
    for box, label in zip(bboxes, labels):
        if box[-1] < score_thr:
            continue
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
        box_int = [int(x) for x in box[:4]]
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        label_text += '|{:.02f}'.format(box[-1])
        cv2.putText(img1, label_text, (box_int[0], box_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    plt.subplot(1, 1, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.show()
    cv2.waitKey(0)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    
    model = init_detector(cfg, args.checkpoint, device='cuda:0')

    # get image list
    if args.dataset == 'VisDrone':
        root_dir = '../data/visdrone2019/detect_voc'
        classes = VisDroneDataset.CLASSES
    elif args.dataset == 'HKB':
        root_dir = '/home/twsf/data/HKB/detect_voc'
        classes = HKBDataset.CLASSES

    list_file = os.path.join(root_dir, 'ImageSets/Main/test.txt')
    image_dir = os.path.join(root_dir, 'JPEGImages')
    with open(list_file, 'r') as f:
        images = [x.strip() for x in f.readlines()]
    imglist = [os.path.join(image_dir, x+'.jpg') for x in images]

    if False:
        imglist = glob.glob('samples/*.jpg')

    results = []
    t0 = time.time()
    for i, img in enumerate(imglist):
        preds = inference_detector(model, img)
        detect_time = time.time() - t0
        sys.stdout.write('\r im_detect: {:d}/{:d} {:s} {:.3f}s'
                        .format(i + 1, len(imglist), imglist[i].split('/')[-1], detect_time))
        sys.stdout.flush()

        if args.show:
            show_result(imglist[i], preds, classes, 0.4)
        else:
            img_id = os.path.basename(imglist[i])
            boxes = np.vstack(preds)
            if boxes.shape[0] == 0:
                continue
            boxes[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(preds)
            ]
            labels = np.concatenate(labels)
            for box, label in zip(boxes, labels):
                results.append({'image_id': img_id,
                                'category_id': label,
                                'bbox': box[:4],
                                'score': box[4]})
        t0 = time.time()

    if not args.show:
        result_file = 'results_region_%s.json' % args.dataset.lower()
        with open(result_file, 'w') as f:
            json.dump(results, f, cls=MyEncoder)
            print('results json saved.')


if __name__ == '__main__':
    main()

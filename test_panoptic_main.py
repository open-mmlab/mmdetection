from copy import deepcopy
import tempfile
import os.path as osp
import os
import mmcv
import numpy as np
import torch
from mmengine.fileio import dump
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.evaluation.metrics.coco_panoptic_metric_mmeval import CocoPanopticMetricMMEval
from mmdet.apis import inference_detector, init_detector
from PIL import Image



try:
    import panopticapi
except ImportError:
    panopticapi = None

tmp_dir = tempfile.TemporaryDirectory()
gt_json_path = osp.join(tmp_dir.name, 'gt.json')
gt_seg_dir = osp.join(tmp_dir.name, 'gt_seg')
os.mkdir(gt_seg_dir)


def _create_panoptic_gt_annotations(ann_file, seg_map_dir):
    categories = [{
        'id': 0,
        'name': 'person',
        'supercategory': 'person',
        'isthing': 1
    }, {
        'id': 1,
        'name': 'cat',
        'supercategory': 'cat',
        'isthing': 1
    }, {
        'id': 2,
        'name': 'dog',
        'supercategory': 'dog',
        'isthing': 1
    }, {
        'id': 3,
        'name': 'wall',
        'supercategory': 'wall',
        'isthing': 0
    }]

    images = [{
        'id': 0,
        'width': 80,
        'height': 60,
        'file_name': 'fake_name1.jpg',
    }]

    annotations = [{
        'segments_info': [{
            'id': 1,
            'category_id': 0,
            'area': 400,
            'bbox': [10, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 2,
            'category_id': 0,
            'area': 400,
            'bbox': [30, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 3,
            'category_id': 2,
            'iscrowd': 0,
            'bbox': [50, 10, 10, 5],
            'area': 50
        }, {
            'id': 4,
            'category_id': 3,
            'iscrowd': 0,
            'bbox': [0, 0, 80, 60],
            'area': 3950
        }],
        'file_name': 'fake_name1.png',
        'image_id': 0
    }]

    gt_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 4 is the id of the background class annotation.
    gt = np.zeros((60, 80), dtype=np.int64) + 4
    gt_bboxes = np.array(
        [[10, 10, 10, 40], [30, 10, 10, 40], [50, 10, 10, 5]],
        dtype=np.int64)
    for i in range(3):
        x, y, w, h = gt_bboxes[i]
        gt[y:y + h, x:x + w] = i + 1  # id starts from 1

    rgb_gt_seg_map = np.zeros(gt.shape + (3, ), dtype=np.uint8)
    rgb_gt_seg_map[:, :, 2] = gt // (256 * 256)
    rgb_gt_seg_map[:, :, 1] = gt % (256 * 256) // 256
    rgb_gt_seg_map[:, :, 0] = gt % 256

    img_path = osp.join(seg_map_dir, 'fake_name1.png')
    mmcv.imwrite(rgb_gt_seg_map[:, :, ::-1], img_path)
    dump(gt_json, ann_file)

    return gt_json


def _create_panoptic_data_samples():
    # predictions
    # TP for background class, IoU=3576/4324=0.827
    # ３ the category id of the background class
    pred = np.zeros((60, 80), dtype=np.int64) + 4 * INSTANCE_OFFSET + 3
    pred_bboxes = np.array(
        [
            [11, 11, 10, 40],  # TP IoU=351/449=0.78
            [38, 10, 10, 40],  # FP
            [51, 10, 10, 5]  # TP IoU=45/55=0.818
        ],
        dtype=np.int64)
    pred_labels = np.array([0, 0, 1], dtype=np.int64)   # 对于背景不pred
    for i in range(len(pred_bboxes)):  # i代表instance
        x, y, w, h = pred_bboxes[i]
        pred[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + pred_labels[i]

    data_samples = [{
        'img_id':
        0,
        'ori_shape': (60, 80),
        'img_path':
        'xxx/fake_name1.jpg',
        'segments_info': [{
            'id': 1,
            'category': 0,
            'is_thing': 1
        }, {
            'id': 2,
            'category': 0,
            'is_thing': 1
        }, {
            'id': 3,
            'category': 1,
            'is_thing': 1
        }, {
            'id': 4,
            'category': 3,
            'is_thing': 0
        }],
        'seg_map_path':
        osp.join(gt_seg_dir, 'fake_name1.png'),
        'pred_panoptic_seg': {
            'sem_seg': torch.from_numpy(pred).unsqueeze(0)
        },
    }]

    return data_samples


def test_evaluate_without_json(data_samples, dataset_meta):
    # with tmpfile, without json
    metric = CocoPanopticMetricMMEval(
        ann_file=None,
        seg_prefix=gt_seg_dir,  # 放groundtruth的位置
        classwise=False,
        nproc=1,
        outfile_prefix=None)

    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)
    # assertDictEqual(eval_results, target)

    # without tmpfile and json
    outfile_prefix = f'{tmp_dir.name}/test'
    metric = CocoPanopticMetricMMEval(
        ann_file=None,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        outfile_prefix=outfile_prefix)

    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)
    # assertDictEqual(eval_results, target)


def test_evaluate_with_json(data_samples, dataset_meta):
    # with tmpfile and json
    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        outfile_prefix=None)

    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)
    # assertDictEqual(eval_results, target)

    # classwise
    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=True,
        nproc=1,
        outfile_prefix=None)
    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)
    # assertDictEqual(eval_results, self.target)

    # without tmpfile, with json
    outfile_prefix = f'{tmp_dir.name}/test1'
    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        outfile_prefix=outfile_prefix)
    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)
    # assertDictEqual(eval_results, target)


def test_format_only(data_samples, dataset_meta):

    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        format_only=True,
        outfile_prefix=None)

    outfile_prefix = f'{tmp_dir.name}/test'
    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=1,
        format_only=True,
        outfile_prefix=outfile_prefix)
    metric.dataset_meta = dataset_meta
    metric.process({}, deepcopy(data_samples))
    eval_results = metric.evaluate(size=1)


def test_with_dataset_meta():
    config_name = 'D:\\mmdetection\\configs\\panoptic_fpn\\panoptic-fpn_r50_fpn_1x_coco.py'
    checkpoint = 'D:\\mmdetection\\checkpoints\\panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth'
    gt_json_path = 'D:\\mmeval_ezp\\mmeval_exp\\ground_truth_panoptic.json'
    gt_seg_dir = 'D:\\mmeval_ezp\\mmeval_exp'
    metric = CocoPanopticMetricMMEval(
        ann_file=gt_json_path,
        seg_prefix=gt_seg_dir,
        classwise=False,
        nproc=32,
        format_only=False,
        outfile_prefix=None)
    cfg = Config.fromfile(config_name)
    register_all_modules()
    model = init_detector(cfg, checkpoint, palette='coco', device='cpu')
    dataset_meta = model.dataset_meta
    metric.dataset_meta = dataset_meta
    imgs = []
    imgs = ['D:\\mmeval_ezp\\mmeval_exp\\000000581781.png']
    # for img_name in os.listdir(img_path):
    #     _img_path = osp.join(img_path, img_name)
    #     imgs.append(_img_path)
    # image = Image.open(_img_path)
    # image_pil = np.array(image, dtype=np.uint32)
    # imgs.append(image_pil)
    dataset = cfg.test_dataloader.dataset
    results = inference_detector(model, imgs)
    for i in range(len(results)):
        data_sample = {}
        # for k in dataset[i].keys():
        #     data_sample[k] = dataset[i][k]

        for k in results[i].keys():
            data_sample[k] = results[i].pred_panoptic_seg

        metric.process({}, deepcopy([data_sample]))
        # self.evaluator.process([data_sample])
        metric.evaluate(1)

    # data_samples = result['data_samples']
    # metric.dataset_meta = dataset_meta
    # metric.process({}, deepcopy(data_samples))
    # eval_results = metric.evaluate(size=1)


if __name__ == '__main__':
    _create_panoptic_gt_annotations(
        gt_json_path, gt_seg_dir)  
    dataset_meta = {
        'classes': ('person', 'cat', 'dog', 'wall'),
        'thing_classes': ('person', 'cat', 'dog'),
        'stuff_classes': ('wall', )
    }
    target = {
        'coco_panoptic/PQ': 67.86874803219071,
        'coco_panoptic/SQ': 80.89770126158936,
        'coco_panoptic/RQ': 83.33333333333334,
        'coco_panoptic/PQ_th': 60.45252075318891,
        'coco_panoptic/SQ_th': 79.9959505972869,
        'coco_panoptic/RQ_th': 75.0,
        'coco_panoptic/PQ_st': 82.70120259019427,
        'coco_panoptic/SQ_st': 82.70120259019427,
        'coco_panoptic/RQ_st': 100.0
    }
    data_samples = _create_panoptic_data_samples()

    test_evaluate_without_json(data_samples, dataset_meta)
    # test_evaluate_with_json(data_samples,dataset_meta)
    # test_format_only(data_samples,dataset_meta)
    # test_with_dataset_meta()

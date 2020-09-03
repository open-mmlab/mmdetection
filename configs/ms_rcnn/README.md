# Mask Scoring R-CNN

## Introduction

```
@inproceedings{huang2019msrcnn,
    title={Mask Scoring R-CNN},
    author={Zhaojin Huang and Lichao Huang and Yongchao Gong and Chang Huang and Xinggang Wang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019},
}
```

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN      | caffe      | 1x      | 4.5      |                |  38.2  | 36.0    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848.log.json) |
| R-50-FPN      | caffe      | 2x      | -        | -              | 38.8   | 36.3    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_20200506_004738.log.json) |
| R-101-FPN     | caffe      | 1x      | 6.5      |                | 40.4   | 37.6    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_20200506_004755.log.json) |
| R-101-FPN     | caffe      | 2x      | -        | -              | 41.1   | 38.1    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_20200506_011134.log.json) |
| R-X101-32x4d  | pytorch    | 2x      | 7.9      | 11.0           | 41.8   | 38.7    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206_100113.log.json) |
| R-X101-64x4d  | pytorch    | 1x      | 11.0     | 8.0            | 43.0   | 39.5    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206_091744.log.json) |
| R-X101-64x4d  | pytorch    | 2x      | 11.0     | 8.0            | 42.6   | 39.5    | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308_012247.log.json) |

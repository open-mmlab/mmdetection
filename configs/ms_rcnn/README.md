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
| R-50-FPN      | caffe      | 1x      |          |                |        |         |  |
| R-50-FPN      | caffe      | 2x      |          |                |        |         |  |
| R-101-FPN     | caffe      | 1x      |          |                |        |         |  |
| R-101-FPN     | caffe      | 2x      |          |                |        |         |  |
| R-X101-32x4d  | pytorch    | 2x      | 7.9      | 8.9            | 41.8   | 38.7    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206_100113.log.json) |
| R-X101-64x4d  | pytorch    | 1x      | 11.0     | 7.0            | 43.0   | 39.5    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206_091744.log.json) |
| R-X101-64x4d  | pytorch    | 2x      | 11.0     | 7.1            | 42.6   | 39.5    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308_012247.log.json) |

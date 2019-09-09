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

| Backbone      | style      | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:-------------:|:----------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN      | caffe      | 1x      | 4.3      | 0.537               | 10.1           | 37.4   | 35.5    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth) |
| R-50-FPN      | caffe      | 2x      | -        | -                   | -              | 38.2   | 35.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r50_caffe_fpn_2x_20190525-a07be31e.pth) |
| R-101-FPN     | caffe      | 1x      | 6.2      | 0.682               |  9.1           | 39.8   | 37.2    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_r101_caffe_fpn_1x_20190624-677a5548.pth) |
| R-101-FPN     | caffe      | 2x      | -        | -                   |  -             | 40.7   | 37.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r101_caffe_fpn_2x_20190525-4aee1528.pth) |
| R-X101-32x4d  | pytorch    | 2x      | 7.6      | 0.844               |  8.0           | 41.7   | 38.5    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_x101_32x4d_fpn_2x_20190628-ab454d07.pth) |
| R-X101-64x4d  | pytorch    | 1x      | 10.5     | 1.214               |  6.4           | 42.0   | 39.1    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_x101_64x4d_fpn_1x_20190628-dec32bda.pth) |
| R-X101-64x4d  | pytorch    | 2x      | -       | -                    |  -             | 42.2   | 38.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_x101_64x4d_fpn_2x_20190525-c044c25a.pth) |

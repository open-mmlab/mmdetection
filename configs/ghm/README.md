# Gradient Harmonized Single-stage Detector

## Introduction

```
@inproceedings{li2019gradient,
  title={Gradient Harmonized Single-stage Detector},
  author={Li, Buyu and Liu, Yu and Wang, Xiaogang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50-FPN     | pytorch |   1x    |   4.0    | 2.9            |  37.0  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130_004213.log.json) |
|    R-101-FPN    | pytorch |   1x    |   6.0    | 4.0            |  39.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130_145259.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.2    | 4.5            |  40.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131_113653.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.3   | 4.7            |  41.4  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131_113723.log.json) |

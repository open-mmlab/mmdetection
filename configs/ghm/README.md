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

|    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :------: |
|    R-50-FPN     | pytorch |   1x    |   3.9    | 0.500               | 9.4            |  36.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ghm/retinanet_ghm_r50_fpn_1x_20190608-b9aa5862.pth) |
|    R-101-FPN    | pytorch |   1x    |   5.8    | 0.625               | 8.5            |  39.0  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ghm/retinanet_ghm_r101_fpn_1x_20190608-b885b74a.pth) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    | 0.818               | 7.6            |  40.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ghm/retinanet_ghm_x101_32x4d_fpn_1x_20190608-ed295d22.pth) |
| X-101-64x4d-FPN | pytorch |   1x    |   9.9    | 1.191               | 6.1            |  41.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ghm/retinanet_ghm_x101_64x4d_fpn_1x_20190608-7f2037ce.pth) |
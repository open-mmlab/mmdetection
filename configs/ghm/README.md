# Gradient Harmonized Single-stage Detector

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{li2019gradient,
  title={Gradient Harmonized Single-stage Detector},
  author={Li, Buyu and Liu, Yu and Wang, Xiaogang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    |   4.0    | 3.3            |  37.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130_004213.log.json) |
|    R-101-FPN    | pytorch |   1x    |   6.0    | 4.4            |  39.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130_145259.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.2    | 5.1            |  40.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131_113653.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.3   | 5.2            |  41.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131_113723.log.json) |

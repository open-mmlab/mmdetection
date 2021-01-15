# Libra R-CNN: Towards Balanced Learning for Object Detection

## Introduction

[ALGORITHM]

We provide config files to reproduce the results in the CVPR 2019 paper [Libra R-CNN](https://arxiv.org/pdf/1904.02701.pdf).

```
@inproceedings{pang2019libra,
  title={Libra R-CNN: Towards Balanced Learning for Object Detection},
  author={Pang, Jiangmiao and Chen, Kai and Shi, Jianping and Feng, Huajun and Ouyang, Wanli and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Architecture | Backbone        | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50-FPN        | pytorch | 1x      | 4.6      | 19.0           | 38.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
| Fast R-CNN   | R-50-FPN        | pytorch | 1x      |          |                |        | |
| Faster R-CNN | R-101-FPN       | pytorch | 1x      | 6.5      | 14.4           | 40.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203_001405.log.json) |
| Faster R-CNN | X-101-64x4d-FPN | pytorch | 1x      | 10.8     | 8.5            | 42.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315_231625.log.json) |
| RetinaNet    | R-50-FPN        | pytorch | 1x      | 4.2      | 17.7           | 37.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205_112757.log.json) |

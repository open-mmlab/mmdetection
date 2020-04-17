# Libra R-CNN: Towards Balanced Learning for Object Detection

## Introduction

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

| Architecture | Backbone        | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:------------:|:---------------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| Faster R-CNN | R-50-FPN        | pytorch | 1x      |          |                     |                |        | [model]() &#124; [log]()|
| Fast R-CNN   | R-50-FPN        | pytorch | 1x      |          |                     |                |        | [model]() &#124; [log]() |
| Faster R-CNN | R-101-FPN       | pytorch | 1x      |          |                     |                |        | [model]() &#124; [log]() |
| Faster R-CNN | X-101-64x4d-FPN | pytorch | 1x      |          |                     |                |        | [model]() &#124; [log]() |
| RetinaNet    | R-50-FPN        | pytorch | 1x      | 4.2      | -                   |                | 37.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205_112757.log.json) |

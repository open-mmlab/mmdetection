# Path Aggregation Network for Instance Segmentation

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{liu2018path,
  author = {Shu Liu and
            Lu Qi and
            Haifang Qin and
            Jianping Shi and
            Jiaya Jia},
  title = {Path Aggregation Network for Instance Segmentation},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```

## Results and Models

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN      | pytorch    | 1x      | 4.0      | 17.2           | 37.5   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_20200503_105836.log.json) |

# Region Proposal by Guided Anchoring

## Introduction

We provide config files to reproduce the results in the CVPR 2019 paper for [Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278).

```
@inproceedings{wang2019region,
    title={Region Proposal by Guided Anchoring},
    author={Jiaqi Wang and Kai Chen and Shuo Yang and Chen Change Loy and Dahua Lin},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR 1000 | Download |
|:------:|:---------------:|:-------:|:-------:|:--------:|:--------------:|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
| GA-RPN |    R-50-FPN     |  caffe  |   1x    |          |                |         |    |
| GA-RPN |    R-101-FPN    |  caffe  |   1x    |          |                |         |    |
| GA-RPN | X-101-32x4d-FPN | pytorch |   1x    | 8.5      | 9.5            | 70.6    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220_221326.log.json) |
| GA-RPN | X-101-64x4d-FPN | pytorch |   1x    | 7.1      | 7.2            | 71.2    | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225_152704.log.json) |


|     Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:--------------:|:---------------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------------:|
|  GA-Fast RCNN  |    R-50-FPN     |  caffe  |   1x    |          |                |        | |
| GA-Faster RCNN |    R-50-FPN     |  caffe  |   1x    |          |                |        | |
| GA-Faster RCNN |    R-101-FPN    |  caffe  |   1x    |          |                |        | |
| GA-Faster RCNN | X-101-32x4d-FPN | pytorch |   1x    |          |                |        | |
| GA-Faster RCNN | X-101-64x4d-FPN | pytorch |   1x    |          |                |        | |
|  GA-RetinaNet  | R-50-FPN        |  caffe  |   1x    |          |                |        | |
|  GA-RetinaNet  | R-101-FPN       |  caffe  |   1x    |          |                |        | |
|  GA-RetinaNet  | X-101-32x4d-FPN | pytorch |   1x    |   6.9    |      9.5       |  40.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219_223025.log.json) |
|  GA-RetinaNet  | X-101-64x4d-FPN | pytorch |   1x    |   9.9    |      7.2       |  41.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226_221123.log.json) |



- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring.

- Performance on COCO test-dev benchmark are shown as follows.


|     Method     | Backbone  | Style | Lr schd | Aug Train | Score thr |  AP  | AP_50 | AP_75 | AP_small | AP_medium | AP_large | Download |
|:--------------:|:---------:|:-----:|:-------:|:---------:|:---------:|:----:|:-----:|:-----:|:--------:|:---------:|:--------:|:--------:|
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.05    |      |       |       |          |           |          |  |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.001   |      |       |       |          |           |          |  |
|  GA-RetinaNet  | R-101-FPN | caffe |   1x    |     F     |   0.05    |      |       |       |          |           |          |  |
|  GA-RetinaNet  | R-101-FPN | caffe |   2x    |     T     |   0.05    |      |       |       |          |           |          |  |

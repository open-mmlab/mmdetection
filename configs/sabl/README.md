# Side-Aware Boundary Localization for More Precise Object Detection

## Introduction

We provide config files to reproduce the object detection results in the ECCV 2020 Spotlight paper for [Side-Aware Boundary Localization for More Precise Object Detection](https://arxiv.org/abs/1912.04260).

```
@inproceedings{Wang_2020_ECCV,
    title = {Side-Aware Boundary Localization for More Precise Object Detection},
    author = {Wang, Jiaqi and Zhang, Wenwei and Cao, Yuhang and Chen, Kai and Pang, Jiangmiao and Gong, Tao and Shi, Jianping, Loy, Chen Change and Lin, Dahua},
    booktitle = {ECCV},
    year = {2020}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).


|       Method       | Backbone  | Lr schd | ms-train | box AP |                                                                                                                                                        Download                                                                                                                                                         |
| :----------------: | :-------: | :-----: | :------: | :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL Faster R-CNN  | R-50-FPN  |   1x    |    N     |  39.8  |    [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/sabl_faster_rcnn_r50_fpn_1x_coco-6e7fef5b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/20200819_171645.log.json)    |
| SABL Faster R-CNN  | R-101-FPN |   1x    |    N     |  41.8  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/sabl_faster_rcnn_r101_fpn_1x_coco-cf49b6a1.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/20200819_172325.log.json)   |
| SABL Cascade R-CNN | R-50-FPN  |   1x    |    N     |  41.6  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/sabl_cascade_rcnn_r50_fpn_1x_coco-b47dd570.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/20200819_171640.log.json)   |
| SABL Cascade R-CNN | R-101-FPN |   1x    |    N     |  43.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/sabl_cascade_rcnn_r101_fpn_1x_coco-22cd86a6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/20200819_172617.log.json) |

|     Method     | Backbone  |  GN   | Lr schd |  ms-train   | box AP |                                                                                                                                                                         Download                                                                                                                                                                         |
| :------------: | :-------: | :---: | :-----: | :---------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL RetinaNet | R-50-FPN  |   N   |   1x    |      N      |  37.5  |                       [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-ff40e8f4.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/20200818_164530.log.json)                        |
| SABL RetinaNet | R-50-FPN  |   Y   |   1x    |      N      |  38.8  |                   [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r50_fpn_gn_1x_coco-ad2dbe82.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/20200820_184643.log.json)                   |
| SABL RetinaNet | R-101-FPN |   N   |   1x    |      N      |  39.7  |                      [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/sabl_retinanet_r101_fpn_1x_coco-48c106ca.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/20200820_155238.log.json)                      |
| SABL RetinaNet | R-101-FPN |   Y   |   1x    |      N      |  40.3  |                 [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_1x_coco-bf1183c4.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/20200820_160310.log.json)                  |
| SABL RetinaNet | R-101-FPN |   Y   |   2x    | Y (640~800) |  42.8  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-b54c988f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/20200819_193625.log.json) |
| SABL RetinaNet | R-101-FPN |   Y   |   2x    | Y (480~960) |  43.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-21cbce7b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/20200819_223557.log.json) |

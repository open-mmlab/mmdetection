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

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 |                                                                    Download                                                                    |
| :----: | :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| GA-RPN |    R-50-FPN     |  caffe  |   1x    |   5.0    |        0.55         |      13.3      |  68.5   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_20190513-95e91886.pth)  |
| GA-RPN |    R-101-FPN    |  caffe  |   1x    |   7.1    |        0.66         |      9.8       |  69.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_20190523-91e0b817.pth) |
| GA-RPN | X-101-32x4d-FPN | pytorch |   1x    |   8.5    |        0.88         |      8.5       |  70.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_20190523-a60df28c.pth) |
| GA-RPN | X-101-64x4d-FPN | pytorch |   1x    |   11.4   |        1.24         |      6.5       |  70.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_20190523-9f2449ba.pth) |


|     Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                                                                       Download                                                                       |
| :------------: | :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
|  GA-Fast RCNN  |    R-50-FPN     |  caffe  |   1x    |   3.3    |        0.23         |      14.9      |  39.5  |    [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_fast_r50_caffe_fpn_1x_20190513-c5af9f8b.pth)    |
| GA-Faster RCNN |    R-50-FPN     |  caffe  |   1x    |   5.1    |        0.64         |      9.6       |  39.9  |   [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_faster_r50_caffe_fpn_1x_20190513-a52b31fa.pth)   |
| GA-Faster RCNN |    R-101-FPN    |  caffe  |   1x    |   7.3    |        0.75         |      8.0       |  41.5  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_faster_r101_caffe_fpn_1x_20190523-9a711ec5.pth)   |
| GA-Faster RCNN | X-101-32x4d-FPN | pytorch |   1x    |   8.7    |        0.97         |      7.1       |  42.9  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_20190523-8dc3e59a.pth)   |
| GA-Faster RCNN | X-101-64x4d-FPN | pytorch |   1x    |   11.6   |        1.33         |      5.7       |  43.9  |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_20190523-a8589c97.pth)   |
|  GA-RetinaNet  |    R-50-FPN     |  caffe  |   1x    |   3.2    |        0.50         |      10.7      |  37.0  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_20190513-29905101.pth)  |
|  GA-RetinaNet  |    R-101-FPN    |  caffe  |   1x    |   5.3    |        0.63         |      8.5       |  38.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_20190523-792ad63d.pth) |
|  GA-RetinaNet  | X-101-32x4d-FPN | pytorch |   1x    |   6.7    |        0.87         |      7.5       |  40.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_20190523-4ec3f13c.pth) |
|  GA-RetinaNet  | X-101-64x4d-FPN | pytorch |   1x    |   9.6    |        1.22         |      5.8       |  40.8  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_20190523-013d1913.pth) |



- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring.
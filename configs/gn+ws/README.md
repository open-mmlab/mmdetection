# Weight Standardization

## Introduction

```
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
```

## Results and Models

Faster R-CNN

| Backbone  | Style   | Normalization | Lr schd | box AP | mask AP | Download |
|:---------:|:-------:|:-------------:|:-------:|:------:|:-------:|:--------:|
| R-50-FPN  | pytorch | GN            | 1x      | 37.8   | -       | - |
| R-50-FPN  | pytorch | GN+WS         | 1x      | 38.9   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/faster_rcnn_r50_fpn_gn_ws_1x_20190418-935d00b6.pth) |
| R-101-FPN | pytorch | GN            | 1x      | 39.8   | -       | - |
| R-101-FPN | pytorch | GN+WS         | 1x      | 41.4   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/faster_rcnn_r101_fpn_gn_ws_1x_20190419-728705ec.pth) |
| X-50-32x4d-FPN | pytorch | GN       | 1x      | 36.5   | -       | - |
| X-50-32x4d-FPN | pytorch | GN+WS    | 1x      | 39.9   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/faster_rcnn_x50_32x4d_fpn_gn_ws_1x_20190419-4e61072b.pth) |
| X-101-32x4d-FPN | pytorch | GN      | 1x      | 33.2   | -       | - |
| X-101-32x4d-FPN | pytorch | GN+WS   | 1x      | 41.8   | -       | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/faster_rcnn_x101_32x4d_fpn_gn_ws_1x_20190419-c78e5583.pth) |

Mask R-CNN

| Backbone  | Style   | Normalization | Lr schd | box AP | mask AP | Download |
|:---------:|:-------:|:-------------:|:-------:|:------:|:-------:|:--------:|
| R-50-FPN  | pytorch | GN            | 2x      | 39.9   | 36.0    | - |
| R-50-FPN  | pytorch | GN+WS         | 2x      | 40.3   | 36.2    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r50_fpn_gn_ws_2x_20190419-9ec97bbb.pth) |
| R-101-FPN | pytorch | GN            | 2x      | 41.6   | 37.3    | - |
| R-101-FPN | pytorch | GN+WS         | 2x      | 42.0   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r101_fpn_gn_ws_2x_20190419-bc7399a6.pth) |
| X-50-32x4d-FPN | pytorch | GN       | 2x      | 39.2   | 35.5    | - |
| X-50-32x4d-FPN | pytorch | GN+WS    | 2x      | 40.7   | 36.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_x50_32x4d_fpn_gn_ws_2x_20190419-2110205e.pth) |
| X-101-32x4d-FPN | pytorch | GN      | 2x      | 36.4   | 33.1    | - |
| X-101-32x4d-FPN | pytorch | GN+WS   | 2x      | 42.1   | 37.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_x101_32x4d_fpn_gn_ws_2x_20190419-7777b15f.pth) |
| R-50-FPN  | pytorch | GN            | 20-23-24e | 40.6   | 36.6    | - |
| R-50-FPN  | pytorch | GN+WS         | 20-23-24e | 41.1   | 37.0    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r50_fpn_gn_ws_20_23_24e_20190425-1d9e499e.pth) |
| R-101-FPN | pytorch | GN            | 20-23-24e | 42.3   | 38.1    | - |
| R-101-FPN | pytorch | GN+WS         | 20-23-24e | 43.0   | 38.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r101_fpn_gn_ws_20_23_24e_20190425-66cb3792.pth) |
| X-50-32x4d-FPN | pytorch | GN       | 20-23-24e | 39.6   | 35.9    | - |
| X-50-32x4d-FPN | pytorch | GN+WS    | 20-23-24e | 41.9   | 37.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_x50_32x4d_fpn_gn_ws_20_23_24e_20190425-d01e2200.pth) |
| X-101-32x4d-FPN | pytorch | GN      | 20-23-24e | 36.6   | 33.4    | - |
| X-101-32x4d-FPN | pytorch | GN+WS   | 20-23-24e | 43.4   | 38.7    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_x101_32x4d_fpn_gn_ws_20_23_24e_20190425-1ff3e5b2.pth) |

Note:

- GN+WS requires about 5% more memory than GN, and it is only 5% slower than GN.
- In the paper, a 20-23-24e lr schedule is used instead of 2x.
- The X-50-GN and X-101-GN pretrained models are also shared by the authors.
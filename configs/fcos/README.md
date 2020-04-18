# FCOS: Fully Convolutional One-Stage Object Detection

## Introduction

```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```

## Results and Models

| Backbone  | Style   | GN      | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | N       | N        | 1x      |          |                |        | |
| R-50      | caffe   | Y       | N        | 1x      |          |                |        | |
| R-50      | caffe   | Y       | N        | 2x      |          |                |        | |
| R-101     | caffe   | Y       | N        | 1x      |          |                |        | |
| R-101     | caffe   | Y       | N        | 2x      |          |                |        | |


| Backbone  | Style   | GN      | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | Y       | Y        | 2x      |          |                |        |  |
| R-101     | caffe   | Y       | Y        | 2x      |          |                |        |  |
| X-101     | pytorch | Y       | Y        | 2x      | 10.0     | 8.6            | 42.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229_222104.log.json) |

**Notes:**
- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU for R-50 and R-101 models, and 8 GPUs with 2 image/GPU for X-101 models.
- The X-101 backbone is X-101-64x4d.

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

| Backbone  | Style   | GN      | MS train | Tricks  | DCN     | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | N       | N        | N       | N       | 1x      | 5.2      | 22.9           | 36.2   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_4x4_1x_coco/fcos_r50_caffe_fpn_1x_4gpu_20200218-c229552f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_4x4_1x_coco/20200224_230410.log.json) |
| R-50      | caffe   | Y       | N        | N       | N       | 1x      | 6.5      | 22.7           | 36.6   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/fcos_r50_caffe_fpn_gn_1x_4gpu_20200218-7831950c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/20200130_004230.log.json) |
| R-50      | caffe   | Y       | N        | Y       | N       | 1x      | -        | -              | 38.6   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco_20200603-67b3859f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco_20200603.log.json)|
| R-50      | caffe   | Y       | N        | Y       | Y       | 1x      | -        | -              | 42.5   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco_20200603-ed16da04.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco_20200603.log.json)|
| R-50      | caffe   | Y       | N        | N       | N       | 2x      | -        | -              | 36.9   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco/fcos_r50_caffe_fpn_gn_2x_4gpu_20200218-8ceb5c76.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco/20200130_004232.log.json) |
| R-101     | caffe   | Y       | N        | N       | N       | 1x      | 10.2     | 17.3           | 39.2   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_1x_coco/fcos_r101_caffe_fpn_gn_1x_4gpu_20200218-13e2cc55.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_1x_coco/20200130_004231.log.json) |
| R-101     | caffe   | Y       | N        | N       | N       | 2x      | -        | -              | 39.1   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_2x_coco/fcos_r101_caffe_fpn_gn_2x_4gpu_20200218-d2261033.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_2x_coco/20200130_004231.log.json) |


| Backbone  | Style   | GN      | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | caffe   | Y       | Y        | 2x      | 6.5      | 22.9           | 38.7   |  [model]() &#124; [log]() |
| R-101     | caffe   | Y       | Y        | 2x      | 10.2     | 17.3           | 40.9   |  [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20200218-d8a4f4cf.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco/20200130_004232.log.json) |
| X-101     | pytorch | Y       | Y        | 2x      | 10.0     | 9.3            | 42.5   | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229_222104.log.json) |

**Notes:**
- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU for R-50 and R-101 models, and 8 GPUs with 2 image/GPU for X-101 models.
- The X-101 backbone is X-101-64x4d.
- Tricks means setting `norm_on_bbox`, `centerness_on_reg`, `center_sampling` as `True`.
- DCN means using `DCNv2` in both backbone and head.

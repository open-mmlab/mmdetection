# VarifocalNet: An IoU-aware Dense Object Detector

## Introduction

<!-- [ALGORITHM] -->

**VarifocalNet (VFNet)** learns to predict the IoU-aware classification score which mixes the object presence confidence and localization accuracy together as the detection score for a bounding box. The learning is supervised by the proposed Varifocal Loss (VFL), based on a new star-shaped bounding box feature representation (the features at nine yellow sampling points). Given the new representation, the object localization accuracy is further improved by refining the initially regressed bounding box. The full paper is available at: [https://arxiv.org/abs/2008.13367](https://arxiv.org/abs/2008.13367).

<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/97464778-4b9ab000-197c-11eb-9283-ab2907ee0252.png" width="600px" />
  <p>Learning to Predict the IoU-aware Classification Score.</p>
</div>

## Citing VarifocalNet

```latex
@article{zhang2020varifocalnet,
  title={VarifocalNet: An IoU-aware Dense Object Detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and S{\"u}nderhauf, Niko},
  journal={arXiv preprint arXiv:2008.13367},
  year={2020}
}
```

## Results and Models

| Backbone     | Style     | DCN     | MS train | Lr schd |Inf time (fps) | box AP (val) | box AP (test-dev) | Config | Download |
|:------------:|:---------:|:-------:|:--------:|:-------:|:-------------:|:------------:|:-----------------:|:------:|:--------:|
| R-50         | pytorch   | N       | N        | 1x      | -          | 41.6         | 41.6              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py) |  [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco.json)|
| R-50         | pytorch   | N       | Y        | 2x      | -          | 44.5         | 44.8              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco.json)|
| R-50         | pytorch   | Y       | Y        | 2x      | -          | 47.8         | 48.0              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| R-101        | pytorch   | N       | N        | 1x      | -          | 43.0         | 43.6              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco.json)|
| R-101        | pytorch   | N       | Y        | 2x      | -          | 46.2         | 46.7              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco.json)|
| R-101        | pytorch   | Y       | Y        | 2x      | -          | 49.0         | 49.2              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| X-101-32x4d  | pytorch   | Y       | Y        | 2x      | -          | 49.7         | 50.0              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| X-101-64x4d  | pytorch   | Y       | Y        | 2x      |  -         | 50.4         | 50.8              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|

**Notes:**

- The MS-train scale range is 1333x[480:960] (`range` mode) and the inference scale keeps 1333x800.
- DCN means using `DCNv2` in both backbone and head.
- Inference time will be updated soon.
- More results and pre-trained models can be found in [VarifocalNet-Github](https://github.com/hyz-xmaster/VarifocalNet)

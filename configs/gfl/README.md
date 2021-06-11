# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection results in the paper [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

```latex
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```

## Results and Models

| Backbone          | Style   | Lr schd | Multi-scale Training| Inf time (fps) | box AP | Config | Download |
|:-----------------:|:-------:|:-------:|:-------------------:|:--------------:|:------:|:------:|:--------:|
| R-50              | pytorch | 1x      | No                  | 19.5           | 40.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244.log.json) |
| R-50              | pytorch | 2x      | Yes                 | 19.5           | 42.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802.log.json) |
| R-101             | pytorch | 2x      | Yes                 | 14.7           | 44.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126.log.json) |
| R-101-dcnv2       | pytorch | 2x      | Yes                 | 12.9           | 47.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002.log.json) |
| X-101-32x4d       | pytorch | 2x      | Yes                 | 12.1           | 45.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002.log.json) |
| X-101-32x4d-dcnv2 | pytorch | 2x      | Yes                 | 10.7           | 48.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002.log.json) |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2.* \
[4] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*

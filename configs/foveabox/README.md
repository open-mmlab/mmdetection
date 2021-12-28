# FoveaBox: Beyond Anchor-based Object Detector

## Abstract

<!-- [ABSTRACT] -->

We present FoveaBox, an accurate, flexible, and completely anchor-free framework for object detection. While almost all state-of-the-art object detectors utilize predefined anchors to enumerate possible locations, scales and aspect ratios for the search of the objects, their performance and generalization ability are also limited to the design of anchors. Instead, FoveaBox directly learns the object existing possibility and the bounding box coordinates without anchor reference. This is achieved by: (a) predicting category-sensitive semantic maps for the object existing possibility, and (b) producing category-agnostic bounding box for each position that potentially contains an object. The scales of target boxes are naturally associated with feature pyramid representations. In FoveaBox, an instance is assigned to adjacent feature levels to make the model more accurate.We demonstrate its effectiveness on standard benchmarks and report extensive experimental analysis. Without bells and whistles, FoveaBox achieves state-of-the-art single model performance on the standard COCO and Pascal VOC object detection benchmark. More importantly, FoveaBox avoids all computation and hyper-parameters related to anchor boxes, which are often sensitive to the final detection performance. We believe the simple and effective approach will serve as a solid baseline and help ease future research for object detection.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143885497-332d38a7-b492-4f51-b9d2-ef9d4ad4412a.png"/>
</div>

<!-- [PAPER_TITLE: FoveaBox: Beyond Anchor-based Object Detector] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1904.03797] -->

<!-- [ALGORITHM] -->

## Introduction

FoveaBox is an accurate, flexible and completely anchor-free object detection system for object detection framework, as presented in our paper [https://arxiv.org/abs/1904.03797](https://arxiv.org/abs/1904.03797):
Different from previous anchor-based methods, FoveaBox directly learns the object existing possibility and the bounding box coordinates without anchor reference. This is achieved by: (a) predicting category-sensitive semantic maps for the object existing possibility, and (b) producing category-agnostic bounding box for each position that potentially contains an object.

## Main Results

### Results on R50/101-FPN

| Backbone  | Style   |  align  | ms-train| Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | N       | N       | 1x      | 5.6      | 24.1           | 36.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219_223025.log.json) |
| R-50      | pytorch | N       | N       | 2x      | 5.6      | -              | 37.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203_112043.log.json) |
| R-50      | pytorch | Y       | N       | 2x      | 8.1      | 19.4           | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203_134252.log.json) |
| R-50      | pytorch | Y       | Y       | 2x      | 8.1      | 18.3           | 40.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205_112557.log.json) |
| R-101     | pytorch | N       | N       | 1x      | 9.2      | 17.4           | 38.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r101_fpn_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219_011740.log.json) |
| R-101     | pytorch | N       | N       | 2x      | 11.7     | -              | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r101_fpn_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208_202059.log.json) |
| R-101     | pytorch | Y       | N       | 2x      | 11.7     | 14.7           | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208_203337.log.json) |
| R-101     | pytorch | Y       | Y       | 2x      | 11.7     | 14.7           | 42.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208_202124.log.json) |

[1] *1x and 2x mean the model is trained for 12 and 24 epochs, respectively.* \
[2] *Align means utilizing deformable convolution to align the cls branch.* \
[3] *All results are obtained with a single model and without any test time data augmentation.*\
[4] *We use 4 GPUs for training.*

Any pull requests or issues are welcome.

## Citations

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```latex
@article{kong2019foveabox,
  title={FoveaBox: Beyond Anchor-based Object Detector},
  author={Kong, Tao and Sun, Fuchun and Liu, Huaping and Jiang, Yuning and Shi, Jianbo},
  journal={arXiv preprint arXiv:1904.03797},
  year={2019}
}
```

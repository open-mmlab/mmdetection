# FoveaBox: Beyond Anchor-based Object Detector

FoveaBox is an accurate, flexible and completely anchor-free object detection system for object detection framework, as presented in our paper [https://arxiv.org/abs/1904.03797](https://arxiv.org/abs/1904.03797):
Different from previous anchor-based methods, FoveaBox directly learns the object existing possibility and the bounding box coordinates without anchor reference. This is achieved by: (a) predicting category-sensitive semantic maps for the object existing possibility, and (b) producing category-agnostic bounding box for each position that potentially contains an object.

## Main Results
### Results on R50/101-FPN

| Backbone  | Style   |  align  | ms-train| Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50      | pytorch | N       | N       | 1x      | 5.6      | 24.1           | 36.5   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219_223025.log.json) |
| R-50      | pytorch | N       | N       | 2x      | 5.6      | -              | 37.2   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203_112043.log.json) |
| R-50      | pytorch | Y       | N       | 2x      | 8.1      | 19.4           | 37.9   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203_134252.log.json) |
| R-50      | pytorch | Y       | Y       | 2x      | 8.1      | 18.3           | 40.4   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205_112557.log.json) |
| R-101     | pytorch | N       | N       | 1x      | 9.2      | 17.4           | 38.6   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219_011740.log.json) |
| R-101     | pytorch | N       | N       | 2x      | 11.7     | -              | 40.0   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208_202059.log.json) |
| R-101     | pytorch | Y       | N       | 2x      | 11.7     | 14.7           | 40.0   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208_203337.log.json) |
| R-101     | pytorch | Y       | Y       | 2x      | 11.7     | 14.7           | 42.0   | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208_202124.log.json) |

[1] *1x and 2x mean the model is trained for 12 and 24 epochs, respectively.* \
[2] *Align means utilizing deformable convolution to align the cls branch.* \
[3] *All results are obtained with a single model and without any test time data augmentation.*\
[4] *We use 4 GPUs for training.*

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{kong2019foveabox,
  title={FoveaBox: Beyond Anchor-based Object Detector},
  author={Kong, Tao and Sun, Fuchun and Liu, Huaping and Jiang, Yuning and Shi, Jianbo},
  journal={arXiv preprint arXiv:1904.03797},
  year={2019}
}
```

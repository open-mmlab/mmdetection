# RepPoints

> [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490)

<!-- [ALGORITHM] -->

## Abstract

Modern object detectors rely heavily on rectangular bounding boxes, such as anchors, proposals and the final predictions, to represent objects at various recognition stages. The bounding box is convenient to use but provides only a coarse localization of objects and leads to a correspondingly coarse extraction of object features. In this paper, we present RepPoints(representative points), a new finer representation of objects as a set of sample points useful for both localization and recognition. Given ground truth localization and recognition targets for training, RepPoints learn to automatically arrange themselves in a manner that bounds the spatial extent of an object and indicates semantically significant local areas. They furthermore do not require the use of anchors to sample a space of bounding boxes. We show that an anchor-free object detector based on RepPoints can be as effective as the state-of-the-art anchor-based detection methods, with 46.5 AP and 67.4 AP50 on the COCO test-dev detection benchmark, using ResNet-101 model.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143972514-93247220-4dad-4eb3-a51b-a1115dc7d449.png"/>
</div>

## Introdution

By [Ze Yang](https://yangze.tech/), [Shaohui Liu](http://b1ueber2y.me/), and [Han Hu](https://ancientmooner.github.io/).

We provide code support and configuration files to reproduce the results in the paper for
["RepPoints: Point Set Representation for Object Detection"](https://arxiv.org/abs/1904.11490) on COCO object detection.

**RepPoints**, initially described in [arXiv](https://arxiv.org/abs/1904.11490), is a new representation method for visual objects, on which visual understanding tasks are typically centered. Visual object representation, aiming at both geometric description and appearance feature extraction, is conventionally achieved by `bounding box + RoIPool (RoIAlign)`. The bounding box representation is convenient to use; however, it provides only a rectangular localization of objects that lacks geometric precision and may consequently degrade feature quality. Our new representation, RepPoints, models objects by a `point set` instead of a `bounding box`, which learns to adaptively position themselves over an object in a manner that circumscribes the object’s `spatial extent` and enables `semantically aligned feature extraction`. This richer and more flexible representation maintains the convenience of bounding boxes while facilitating various visual understanding applications. This repo demonstrated the effectiveness of RepPoints for COCO object detection.

Another feature of this repo is the demonstration of an `anchor-free detector`, which can be as effective as state-of-the-art anchor-based detection methods. The anchor-free detector can utilize either `bounding box` or `RepPoints` as the basic object representation.

## Results and Models

The results on COCO 2017val are shown in the table below.

|  Method   |   Backbone    | GN  | Anchor | convert func | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                 Config                                  |                                                                                                                                                                                                                 Download                                                                                                                                                                                                                  |
| :-------: | :-----------: | :-: | :----: | :----------: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   BBox    |   R-50-FPN    |  Y  | single |      -       |   1x    |   3.9    |      15.9      |  36.4  |      [config](./reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py)      |                                     [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916-0eedf8d1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916.log.json)                                      |
|   BBox    |   R-50-FPN    |  Y  |  none  |      -       |   1x    |   3.9    |      15.4      |  37.4  |  [config](./reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco.py)   |                                     [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916-0eedf8d1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916.log.json)                                      |
| RepPoints |   R-50-FPN    |  N  |  none  |    moment    |   1x    |   3.3    |      18.5      |  37.0  |             [config](./reppoints-moment_r50_fpn_1x_coco.py)             |                                                         [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330_233609.log.json)                                                         |
| RepPoints |   R-50-FPN    |  Y  |  none  |    moment    |   1x    |   3.9    |      17.5      |  38.1  |       [config](./reppoints-moment_r50_fpn-gn_head-gn_1x_coco.py)        |                       [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329_145952-3e51b550.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329_145952.log.json)                        |
| RepPoints |   R-50-FPN    |  Y  |  none  |    moment    |   2x    |   3.9    |       -        |  38.6  |       [config](./reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py)        |                           [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329-91babaa2.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329_150020.log.json)                           |
| RepPoints |   R-101-FPN   |  Y  |  none  |    moment    |   2x    |   5.8    |      13.7      |  40.5  |       [config](./reppoints-moment_r101_fpn-gn_head-gn_2x_coco.py)       |                         [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329_132205.log.json)                         |
| RepPoints | R-101-FPN-DCN |  Y  |  none  |    moment    |   2x    |   5.9    |      12.1      |  42.9  | [config](./reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-3309fbf2.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329_132134.log.json) |
| RepPoints | X-101-FPN-DCN |  Y  |  none  |    moment    |   2x    |   7.1    |      9.3       |  44.2  | [config](./reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329_132201.log.json) |

**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively.
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `none` in the `anchor` column means 2-d `center point` (x,y) is used to represent the initial object hypothesis. `single` denotes one 4-d anchor box (x,y,w,h) with IoU based label assign criterion is adopted.
- `moment`, `partial MinMax`, `MinMax` in the `convert func` column are three functions to convert a point set to a pseudo box.
- Note the results here are slightly different from those reported in the paper, due to framework change. While the original paper uses an [MXNet](https://mxnet.apache.org/) implementation, we re-implement the method in [PyTorch](https://pytorch.org/) based on mmdetection.

## Citation

```latex
@inproceedings{yang2019reppoints,
  title={RepPoints: Point Set Representation for Object Detection},
  author={Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```

# FSAF

> [Feature Selective Anchor-Free Module for Single-Shot Object Detection](https://arxiv.org/abs/1903.00621)

<!-- [ALGORITHM] -->

## Abstract

We motivate and present feature selective anchor-free (FSAF) module, a simple and effective building block for single-shot object detectors. It can be plugged into single-shot detectors with feature pyramid structure. The FSAF module addresses two limitations brought up by the conventional anchor-based detection: 1) heuristic-guided feature selection; 2) overlap-based anchor sampling. The general concept of the FSAF module is online feature selection applied to the training of multi-level anchor-free branches. Specifically, an anchor-free branch is attached to each level of the feature pyramid, allowing box encoding and decoding in the anchor-free manner at an arbitrary level. During training, we dynamically assign each instance to the most suitable feature level. At the time of inference, the FSAF module can work jointly with anchor-based branches by outputting predictions in parallel. We instantiate this concept with simple implementations of anchor-free branches and online feature selection strategy. Experimental results on the COCO detection track show that our FSAF module performs better than anchor-based counterparts while being faster. When working jointly with anchor-based branches, the FSAF module robustly improves the baseline RetinaNet by a large margin under various settings, while introducing nearly free inference overhead. And the resulting best model can achieve a state-of-the-art 44.6% mAP, outperforming all existing single-shot detectors on COCO.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143887228-c790b542-c0f7-4113-8597-12250c50fe8f.png"/>
</div>

## Introduction

FSAF is an anchor-free method published in CVPR2019 ([https://arxiv.org/pdf/1903.00621.pdf](https://arxiv.org/pdf/1903.00621.pdf)).
Actually it is equivalent to the anchor-based method with only one anchor at each feature map position in each FPN level.
And this is how we implemented it.
Only the anchor-free branch is released for its better compatibility with the current framework and less computational budget.

In the original paper, feature maps within the central 0.2-0.5 area of a gt box are tagged as ignored. However,
it is empirically found that a hard threshold (0.2-0.2) gives a further gain on the performance. (see the table below)

## Results and Models

### Results on R50/R101/X101-FPN

| Backbone | ignore range | ms-train | Lr schd | Train Mem (GB) | Train time (s/iter) | Inf time (fps) |   box AP    |                                                   Config                                                    |                                                                                                                                                                    Download                                                                                                                                                                     |
| :------: | :----------: | :------: | :-----: | :------------: | :-----------------: | :------------: | :---------: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |   0.2-0.5    |    N     |   1x    |      3.15      |        0.43         |      12.3      | 36.0 (35.9) |                                                                                                             | [model](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715-b555b0e0.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715_094657.log.json) |
|   R-50   |   0.2-0.2    |    N     |   1x    |      3.15      |        0.43         |      13.0      |    37.4     |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py)     |                                             [model](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco_20200428_072327.log.json)                                              |
|  R-101   |   0.2-0.2    |    N     |   1x    |      5.08      |        0.58         |      10.8      | 39.3 (37.9) |    [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r101_fpn_1x_coco.py)    |                                           [model](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco-9e71098f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco_20200428_160348.log.json)                                            |
|  X-101   |   0.2-0.2    |    N     |   1x    |      9.38      |        1.23         |      5.6       | 42.4 (41.0) | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py) |                               [model](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco_20200428_160424.log.json)                                |

**Notes:**

- *1x means the model is trained for 12 epochs.*
- *AP values in the brackets represent those reported in the original paper.*
- *All results are obtained with a single model and single-scale test.*
- *X-101 backbone represents ResNext-101-64x4d.*
- *All pretrained backbones use pytorch style.*
- *All models are trained on 8 Titan-XP gpus and tested on a single gpu.*

## Citation

BibTeX reference is as follows.

```latex
@inproceedings{zhu2019feature,
  title={Feature Selective Anchor-Free Module for Single-Shot Object Detection},
  author={Zhu, Chenchen and He, Yihui and Savvides, Marios},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={840--849},
  year={2019}
}
```

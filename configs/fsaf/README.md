# Feature Selective Anchor-Free Module for Single-Shot Object Detection

FSAF is an anchor-free method published in CVPR2019 ([https://arxiv.org/pdf/1903.00621.pdf](https://arxiv.org/pdf/1903.00621.pdf)).
Actually it is equivalent to the anchor-based method with only one anchor at each feature map position in each FPN level.
And this is how we implemented it.
Only the anchor-free branch is released for its better compatibility with the current framework and less computational budget.

In the original paper, feature maps within the central 0.2-0.5 area of a gt box are tagged as ignored. However,
it is empirically found that a hard threshold (0.2-0.2) gives a further gain on the performance. (see the table below)

## Main Results
### Results on R50/R101/X101-FPN

| Backbone   |  ignore range | ms-train| Lr schd |Train Mem (GB)| Train time (s/iter) | Inf time (fps) | box AP | Download |
|:----------:|  :-------:    |:-------:|:-------:|:------------:|:---------------:|:--------------:|:-------------:|:--------:|
| R-50       |   0.2-0.5     | N       | 1x      |    3.15      | 0.43            |    12.3        | 37.0 (35.9)   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco-9ad4c643.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200428_122907.log.json)  |
| R-50       |   0.2-0.2     | N       | 1x      |    3.15      | 0.43            |    13.0        | 37.4          | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco_20200428_072327.log.json)|
| R-101      |   0.2-0.2     | N       | 1x      |    5.08      | 0.58            |    10.8        | 39.3 (37.9)   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco-9e71098f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco_20200428_160348.log.json)|
| X-101      |   0.2-0.2     | N       | 1x      |    9.38      | 1.23            |    5.6         | 42.4 (41.0)   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco_20200428_160424.log.json)|

**Notes:**
 - *1x means the model is trained for 12 epochs.*
 - *AP values in the brackets represent those reported in the original paper.*
 - *All results are obtained with a single model and single-scale test.*
 - *X-101 backbone represents ResNext-101-64x4d.*
 - *All pretrained backbones use pytorch style.*
 - *All models are trained on 8 Titan-XP gpus and tested on a single gpu.*

## Citations
BibTeX reference is as follows.
```
@inproceedings{zhu2019feature,
  title={Feature Selective Anchor-Free Module for Single-Shot Object Detection},
  author={Zhu, Chenchen and He, Yihui and Savvides, Marios},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={840--849},
  year={2019}
}
```

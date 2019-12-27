# Feature Selective Anchor-Free Module for Single-Shot Object Detection

FSAF is an anchor-free method published in CVPR2019 ([https://arxiv.org/pdf/1903.00621.pdf](https://arxiv.org/pdf/1903.00621.pdf)).
Actually it is equivalent to the anchor-based method with only one anchor at each feature map position in each FPN level.
And this is how we implemented it.
Only the anchor-free branch is released for its better compatibility with the current framework and less computational budget.

In the original paper, feature maps within the central 0.2-0.5 area of a gt box are tagged as ignored. However,
it is empirically found that a hard threshold (0.2-0.2) gives a further gain on the performance. (see the table below)

## Main Results
### Results on R50/R101/X101-FPN

| Backbone   |  ignore range | ms-train| Lr schd | Train Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:----------:|  :-------:    |:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50       |   0.2-0.5     | N       | 1x      |    2.97      | 0.43            |    12.3        | 35.7   | fsaf-r50-fpn-1x-20191226-ee860779ad09031f7a58d193b0438a26.pth  |
| R-50       |   0.2-0.2     | N       | 1x      |    2.97      | 0.43            |    13.0        | 37.0   | fsaf-r50-fpn-1x-20191225-d388a744213c3bb187e073f5ccdde5d6.pth  |
| R-101      |   0.2-0.5     | N       | 1x      |    4.87      | 0.58            |    10.6        | 37.8   | fsaf-r101-fpn-1x-20191226-736730f8db59ac0a28262034484ed57d.pth |
| R-101      |   0.2-0.2     | N       | 1x      |    4.87      | 0.58            |    10.8        | 39.1   | fsaf-r101-fpn-1x-20191225-e1dbbcba40933cd8fc0d0174d1b13aa7.pth |
| X-101      |   0.2-0.2     | N       | 1x      |    9.02      | 1.23            |    5.6         | 41.8   | fsaf-x101-64x4d-fpn-1x-20191225-82d23b4bc07f2d666eed71fe48de49a9.pth |

**Notes:**
 - *1x and 2x mean the model is trained for 12 and 24 epochs, respectively.*
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

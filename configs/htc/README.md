# Hybrid Task Cascade for Instance Segmentation

## Introduction

We provide config files to reproduce the results in the CVPR 2019 paper for [Hybrid Task Cascade](https://arxiv.org/abs/1901.07518).

```
@inproceedings{chen2019hybrid,
  title={Hybrid task cascade for instance segmentation},
  author={Chen, Kai and Pang, Jiangmiao and Wang, Jiaqi and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Shi, Jianping and Ouyang, Wanli and Chen Change Loy and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Dataset

HTC requires COCO and COCO-stuff dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Backbone  | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:-------:|:--------:|
| R-50-FPN  | pytorch | 1x      | 7.4      | 0.936               | 4.1            | 42.1   | 37.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_1x_20190408-878c1712.pth) |
| R-50-FPN  | pytorch | 20e     | -        | -                   | -              | 43.2   | 38.1    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_20e_20190408-c03b7015.pth) |
| R-101-FPN | pytorch | 20e     | 9.3      | 1.051               | 4.0            | 44.9   | 39.4    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r101_fpn_20e_20190408-a2e586db.pth) |
| X-101-32x4d-FPN | pytorch |20e| 5.8      | 0.769               | 3.8            | 46.1   | 40.3    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_32x4d_fpn_20e_20190408-9eae4d0b.pth) |
| X-101-64x4d-FPN | pytorch |20e| 7.5      | 1.120               | 3.5            | 46.9   | 40.8    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth) |

- In the HTC paper and COCO 2018 Challenge, `score_thr` is set to 0.001 for both baselines and HTC.
- We use 8 GPUs with 2 images/GPU for R-50 and R-101 models, and 16 GPUs with 1 image/GPU for X-101 models.
If you would like to train X-101 HTC with 8 GPUs, you need to change the lr from 0.02 to 0.01.

We also provide a powerful HTC with DCN and multi-scale training model. No testing augmentation is used.

| Backbone         | Style   | DCN   | training scales | Lr schd | box AP | mask AP | Download |
|:----------------:|:-------:|:-----:|:---------------:|:-------:|:------:|:-------:|:--------:|
| X-101-64x4d-FPN  | pytorch | c3-c5 | 400~1400        | 20e     | 50.7   | 43.9    | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth) |

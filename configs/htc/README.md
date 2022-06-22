# HTC

> [Hybrid Task Cascade for Instance Segmentation](ttps://arxiv.org/abs/1901.07518)

<!-- [ALGORITHM] -->

## Abstract

Cascade is a classic yet powerful architecture that has boosted performance on various tasks. However, how to introduce cascade to instance segmentation remains an open question. A simple combination of Cascade R-CNN and Mask R-CNN only brings limited gain. In exploring a more effective approach, we find that the key to a successful instance segmentation cascade is to fully leverage the reciprocal relationship between detection and segmentation. In this work, we propose a new framework, Hybrid Task Cascade (HTC), which differs in two important aspects: (1) instead of performing cascaded refinement on these two tasks separately, it interweaves them for a joint multi-stage processing; (2) it adopts a fully convolutional branch to provide spatial context, which can help distinguishing hard foreground from cluttered background. Overall, this framework can learn more discriminative features progressively while integrating complementary features together in each stage. Without bells and whistles, a single HTC obtains 38.4 and 1.5 improvement over a strong Cascade Mask R-CNN baseline on MSCOCO dataset. Moreover, our overall system achieves 48.6 mask AP on the test-challenge split, ranking 1st in the COCO 2018 Challenge Object Detection Task.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143893906-e05acaa6-d46f-4c11-84e8-bb9940a95b44.png"/>
</div>

## Introduction

HTC requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
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

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                                     Config                                                      |                                                                                                                                                   Download                                                                                                                                                    |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   8.2    |      5.8       |  42.3  |  37.4   |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r50_fpn_1x_coco.py)        |                           [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317_070435.log.json)                           |
|    R-50-FPN     | pytorch |   20e   |   8.2    |       -        |  43.3  |  38.3   |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r50_fpn_20e_coco.py)       |                         [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319_070313.log.json)                         |
|    R-101-FPN    | pytorch |   20e   |   10.2   |      5.5       |  44.8  |  39.6   |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r101_fpn_20e_coco.py)       |                       [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317_153107.log.json)                       |
| X-101-32x4d-FPN | pytorch |   20e   |   11.4   |      5.0       |  46.1  |  40.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_32x4d_fpn_16x1_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_32x4d_fpn_16x1_20e_coco/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_32x4d_fpn_16x1_20e_coco/htc_x101_32x4d_fpn_16x1_20e_coco_20200318_034519.log.json) |
| X-101-64x4d-FPN | pytorch |   20e   |   14.5   |      4.4       |  47.0  |  41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318_081711.log.json) |

- In the HTC paper and COCO 2018 Challenge, `score_thr` is set to 0.001 for both baselines and HTC.
- We use 8 GPUs with 2 images/GPU for R-50 and R-101 models, and 16 GPUs with 1 image/GPU for X-101 models.
  If you would like to train X-101 HTC with 8 GPUs, you need to change the lr from 0.02 to 0.01.

We also provide a powerful HTC with DCN and multi-scale training model. No testing augmentation is used.

|    Backbone     |  Style  |  DCN  | training scales | Lr schd | box AP | mask AP |                                                                    Config                                                                    |                                                                                                                                                                                                             Download                                                                                                                                                                                                              |
| :-------------: | :-----: | :---: | :-------------: | :-----: | :----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| X-101-64x4d-FPN | pytorch | c3-c5 |    400~1400     |   20e   |  50.4  |  43.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312_203410.log.json) |

## Citation

We provide config files to reproduce the results in the CVPR 2019 paper for [Hybrid Task Cascade](https://arxiv.org/abs/1901.07518).

```latex
@inproceedings{chen2019hybrid,
  title={Hybrid task cascade for instance segmentation},
  author={Chen, Kai and Pang, Jiangmiao and Wang, Jiaqi and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Shi, Jianping and Ouyang, Wanli and Chen Change Loy and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

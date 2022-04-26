# RegNet

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

<!-- [BACKBONE] -->

## Abstract

 In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143971942-da50f719-61e9-43bd-9468-0dbfbe80284e.png"/>
</div>

## Introduction

We implement RegNetX and RegNetY models in detection systems and provide their first results on Mask R-CNN, Faster R-CNN and RetinaNet.

The pre-trained models are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

## Usage

To use a regnet model, there are two steps to do:

1. Convert the model to ResNet-style supported by MMDetection
2. Modify backbone and neck in config accordingly

### Convert model

We already prepare models of FLOPs from 400M to 12G in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

```bash
python -u tools/model_converters/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

## Results and Models

### Mask R-CNN

|   Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    [R-50-FPN](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.4      | 12.0           | 38.2   | 34.7    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141.log.json)   |
|[RegNetX-4.0GF-FPN](./mask_rcnn_regnetx-4GF_fpn_1x_coco.py)| pytorch |   1x    |5.5||41.5|37.4|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217-32e9c92d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217.log.json)   |
|    [R-101-FPN](../mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py)| pytorch |   1x    | 6.4      | 10.3           | 40.0   | 36.1    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|[RegNetX-6.4GF-FPN](./mask_rcnn_regnetx-6.4GF_fpn_1x_coco.py)| pytorch |   1x    |6.1 ||41.0|37.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco/mask_rcnn_regnetx-6.4GF_fpn_1x_coco_20200517_180439-3a7aae83.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco/mask_rcnn_regnetx-6.4GF_fpn_1x_coco_20200517_180439.log.json)   |
| [X-101-32x4d-FPN](../mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | pytorch |   1x    | 7.6      | 9.4            | 41.9   | 37.5    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
|[RegNetX-8.0GF-FPN](./mask_rcnn_regnetx-8GF_fpn_1x_coco.py)| pytorch |   1x    |6.4 ||41.7|37.5|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515-09daa87e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515.log.json)   |
|[RegNetX-12GF-FPN](./mask_rcnn_regnetx-12GF_fpn_1x_coco.py)| pytorch |   1x    |7.4 ||42.2|38|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552-b538bd8b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552.log.json) |
|[RegNetX-3.2GF-FPN-DCN-C3-C5](./mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726-75f40794.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726.log.json)   |

### Faster R-CNN

|   Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    [R-50-FPN](../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.0      | 18.2           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    | 4.5||39.9|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927-126fd9bf.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927.log.json)   |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_2x_coco.py)| pytorch |   2x    | 4.5||41.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955-e2081918.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955.log.json)   |

### RetinaNet

|  Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|    [R-50-FPN](../retinanet/retinanet_r50_fpn_1x_coco.py)     | pytorch |   1x    |   3.8    |      16.6      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|[RegNetX-800MF-FPN](./retinanet_regnetx-800MF_fpn_1x_coco.py)| pytorch |   1x    |2.5||35.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-800MF_fpn_1x_coco/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403-f6f91d10.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-800MF_fpn_1x_coco/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-1.6GF-FPN](./retinanet_regnetx-1.6GF_fpn_1x_coco.py)| pytorch |   1x    |3.3||37.3|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403-37009a9d.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-3.2GF-FPN](./retinanet_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    |4.2 ||39.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141-cb1509e8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141.log.json)   |

### Pre-trained models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Method   |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-----: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |  :--------: |
|Faster RCNN |[RegNetX-400MF-FPN](./faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |2.3 ||37.1|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210526_095112-e1967c37.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210526_095112.log.json)   |
|Faster RCNN |[RegNetX-800MF-FPN](./faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |2.8 ||38.8|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210526_095118-a2c70b20.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210526_095118.log.json)   |
|Faster RCNN |[RegNetX-1.6GF-FPN](./faster_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |3.4 ||40.5|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-1_20210526_095325-94aa46cc.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-1_20210526_095325.log.json)   |
|Faster RCNN |[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |4.4 ||42.3|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-3_20210526_095152-e16a5227.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-3_20210526_095152.log.json)   |
|Faster RCNN |[RegNetX-4GF-FPN](./faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |4.9 ||42.8|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210526_095201-65eaf841.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210526_095201.log.json)   |
|Mask RCNN |[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.0 ||43.1|38.7|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221-99879813.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221.log.json)   |
|Mask RCNN |[RegNetX-400MF-FPN](./mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco.py)| pytorch |   3x    |2.5 ||37.6|34.4|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco_20210601_235443-8aac57a4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-400MF_fpn_mstrain-poly_3x_coco_20210601_235443.log.json)   |
|Mask RCNN |[RegNetX-800MF-FPN](./mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco.py)| pytorch |   3x    |2.9 ||39.5|36.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco_20210602_210641-715d51f5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-800MF_fpn_mstrain-poly_3x_coco_20210602_210641.log.json)   |
|Mask RCNN |[RegNetX-1.6GF-FPN](./mask_rcnn_regnetx-1.6GF_fpn_mstrain-poly_3x_coco.py)| pytorch |   3x    |3.6 ||40.9|37.5|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-1.6GF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-1_20210602_210641-6764cff5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-1.6GF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-1_20210602_210641.log.json)   |
|Mask RCNN |[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)     | pytorch |   3x    |5.0 ||43.1|38.7|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221-99879813.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221.log.json)   |
|Mask RCNN |[RegNetX-4GF-FPN](./mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco.py)    | pytorch |   3x    |5.1 ||43.4|39.2|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco_20210602_032621-00f0331c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco/mask_rcnn_regnetx-4GF_fpn_mstrain-poly_3x_coco_20210602_032621.log.json)   |
|Cascade Mask RCNN |[RegNetX-400MF-FPN](./cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |4.3||41.6|36.4|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210715_211619-5142f449.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco_20210715_211619.log.json)   |
|Cascade Mask RCNN |[RegNetX-800MF-FPN](./cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |4.8||42.8|37.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210715_211616-dcbd13f4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-800MF_fpn_mstrain_3x_coco_20210715_211616.log.json)   |
|Cascade Mask RCNN |[RegNetX-1.6GF-FPN](./cascade_mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.4||44.5|39.0|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/cascade_mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-1_20210715_211616-75f29a61.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-1.6GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-1_20210715_211616.log.json)   |
|Cascade Mask RCNN |[RegNetX-3.2GF-FPN](./cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |6.4||45.8|40.0|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-3_20210715_211616-b9c2c58b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-3_20210715_211616.log.json)   |
|Cascade Mask RCNN |[RegNetX-4GF-FPN](./cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco.py)    | pytorch |   3x    |6.9||45.8|40.0|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210715_212034-cbb1be4c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/regnet/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco/cascade_mask_rcnn_regnetx-4GF_fpn_mstrain_3x_coco_20210715_212034.log.json)   |

### Notice

1. The models are trained using a different weight decay, i.e., `weight_decay=5e-5` according to the setting in ImageNet training. This brings improvement of at least 0.7 AP absolute but does not improve the model using ResNet-50.
2. RetinaNets using RegNets are trained with learning rate 0.02 with gradient clip. We find that using learning rate 0.02 could improve the results by at least 0.7 AP absolute and gradient clip is necessary to stabilize the training. However, this does not improve the performance of ResNet-50-FPN RetinaNet.

## Citation

```latex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

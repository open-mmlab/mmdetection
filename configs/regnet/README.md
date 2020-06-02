# Designing Network Design Spaces

## Introduction

We implement RegNetX and RegNetY models in detection systems and provide their first results on Mask R-CNN, Faster R-CNN and RetinaNet.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

```
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Usage

To use a regnet model, there are two steps to do:
1. Convert the model to ResNet-style supported by MMDetection
2. Modify backbone and neck in config accordingly

### Convert model

We already prepare models of FLOPs from 800M to 12G in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

```bash
python -u tools/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```
This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.


### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

## Results

### Mask R-CNN
|   Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    [R-50-FPN](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.4      | 12.0           | 38.2   | 34.7    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3GF_fpn_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_1x_coco/mask_rcnn_regnetx-3GF_fpn_1x_coco_20200520_163141-2a9d1814.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_1x_coco/mask_rcnn_regnetx-3GF_fpn_1x_coco_20200520_163141.log.json)   |
|[RegNetX-4.0GF-FPN](./mask_rcnn_regnetx-4GF_fpn_1x_coco.py)| pytorch |   1x    |5.5||41.5|37.4|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217-32e9c92d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217.log.json)   |
|    [R-101-FPN](../mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py)| pytorch |   1x    | 6.4      | 10.3           | 40.0   | 36.1    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|[RegNetX-6.4GF-FPN](./mask_rcnn_regnetx-6GF_fpn_1x_coco.py)| pytorch |   1x    |6.1 ||41.0|37.1|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6GF_fpn_1x_coco/mask_rcnn_regnetx-6GF_fpn_1x_coco_20200517_180439-3a7aae83.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6GF_fpn_1x_coco/mask_rcnn_regnetx-6GF_fpn_1x_coco_20200517_180439.log.json)   |
| [X-101-32x4d-FPN](../mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | pytorch |   1x    | 7.6      | 9.4            | 41.9   | 37.5    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
|[RegNetX-8.0GF-FPN](./mask_rcnn_regnetx-8GF_fpn_1x_coco.py)| pytorch |   1x    |6.4 ||41.7|37.5|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515-09daa87e.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515.log.json)   |
|[RegNetX-12GF-FPN](./mask_rcnn_regnetx-12GF_fpn_1x_coco.py)| pytorch |   1x    |7.4 ||42.2|38|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552-b538bd8b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552.log.json) |
|[RegNetX-3.2GF-FPN-DCN-C3-C5](./mask_rcnn_regnetx-3GF_fpn_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726-75f40794.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726.log.json)   |

### Faster R-CNN
|   Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    [R-50-FPN](../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.0      | 18.2           | 37.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    | 4.5||39.9|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_1x_coco/faster_rcnn_regnetx-3GF_fpn_1x_coco_20200517_175927-126fd9bf.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_1x_coco/faster_rcnn_regnetx-3GF_fpn_1x_coco_20200517_175927.log.json)   |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3GF_fpn_mstrain_2x_coco.py)| pytorch |   2x    | 4.5||41.1|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_2x_coco/faster_rcnn_regnetx-3GF_fpn_2x_coco_20200520_223955-e2081918.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_2x_coco/faster_rcnn_regnetx-3GF_fpn_2x_coco_20200520_223955.log.json)   |

### RetinaNet
|  Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    [R-50-FPN](../retinanet/retinanet_r50_fpn_1x_coco.py)     | pytorch |   1x    |   3.8    |      16.6      |  36.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|[RegNetX-800MF-FPN](./retinanet_regnetx-800MF_fpn_mstrain_1x_coco.py)| pytorch |   1x    |2.5||35.6|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-800MF_fpn_1x_coco/retinanet_r50_regnetx-800MF_fpn_1x_coco_20200517_191403-f6f91d10.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-800MF_fpn_1x_coco/retinanet_r50_regnetx-800MF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-1.6GF-FPN](./retinanet_regnetx-1GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    |3.3||37.3|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-1GF_fpn_1x_coco/retinanet_r50_regnetx-1GF_fpn_1x_coco_20200517_191403-37009a9d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-1GF_fpn_1x_coco/retinanet_r50_regnetx-1GF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-3.2GF-FPN](./retinanet_regnetx-3GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    |4.2 ||39.1|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-3GF_fpn_1x_coco/retinanet_r50_regnetx-3GF_fpn_1x_coco_20200520_163141-cb1509e8.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/retinanet_r50_regnetx-3GF_fpn_1x_coco/retinanet_r50_regnetx-3GF_fpn_1x_coco_20200520_163141.log.json)   |

### Pre-trained models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Method   |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-----: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|Faster RCNN |[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.0 ||42.2|-|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_mstrxin_3x_coco/faster_rcnn_regnetx-3GF_fpn_mstrxin_3x_coco_20200520_224253-bf85ae3e.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3GF_fpn_mstrxin_3x_coco/faster_rcnn_regnetx-3GF_fpn_mstrxin_3x_coco_20200520_224253.log.json)   |
|Mask RCNN |[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.0 ||43.1|38.7|[model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3GF_fpn_mstrain_3x_coco_20200521_202221-99879813.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3GF_fpn_mstrain_3x_coco_20200521_202221.log.json)   |

### Notice
1. The models are trained using a different weight decay, i.e., `weight_decay=5e-5` according to the setting in ImageNet training. This brings improvement of at least 0.7 AP absolute but does not improve the model using ResNet-50.
2. RetinaNets using RegNets are trained with learning rate 0.02 with gradient clip. We find that using learning rate 0.02 could improve the results by at least 0.7 AP absolute and gradient clip is necessary to stabilize the training.
However, this does not improve the performance of ResNet-50-FPN RetinaNet.

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

We already convert most of the models compatible with ResNet-50/101 or ResNeXt-50/101 in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in this directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

```bash
python -u regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```
This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.


### Modify config

For convenience, we list the architecture parameters of pre-trained models maintained by MMDetection as the following.

```json
'RegNetX_3.2GF': {'depth': 25, 'w0': 88, 'wa': 26.31, 'wm': 2.25, 'group_w': 48, 'bot_mul':1.0},
'RegNetX_4.0GF': {'depth': 23, 'w0': 96, 'wa': 38.65, 'wm': 2.43, 'group_w': 40, 'bot_mul': 1.0},
'RegNetX_6.4GF': {'depth': 17, 'w0': 184, 'wa': 60.83, 'wm': 2.07, 'group_w': 56, 'bot_mul': 1.0},
'RegNetX_8.0GF': {'depth': 23, 'w0': 80, 'wa': 49.56, 'wm': 2.88, 'group_w': 120, 'bot_mul': 1.0},
'RegNetX_12GF': {'depth': 19, 'w0': 168, 'wa': 73.36, 'wm': 2.37, 'group_w': 112, 'bot_mul':1.0},
'RegNetX_16GF': {'depth': 22, 'w0': 216, 'wa': 55.59, 'wm': 2.1, 'group_w': 128, 'bot_mul':1.0}
```

The users can modify the config's `depth` of backbone, corresponding keys in `arch_parameter`, and `in_channels` in FPN according to this list.

For self-implemented models, the users are responsible to check these parameters by themselves.

## Results

### Mask R-CNN
|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN*     |  caffe  |   1x    |          |                | 38.8   | 35.4    |   |
|[RegNetX-3.2GF-FPN*](./mask_rcnn_regnetx_3GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    | ||39.5|35.9|   |
|    R-50-FPN     | pytorch |   1x    |          |                | 38.8   | 35.4    |   |
|[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx_3GF_fpn_1x_coco.py)| pytorch |   1x    | ||||   |
|[RegNetX-4.0GF-FPN](./mask_rcnn_regnetx_4GF_fpn_1x_coco.py)| pytorch |   1x    | ||||   |
|    R-101-FPN    | pytorch |   1x    | 6.4      | 10.3           | 40.0   | 36.1    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|[RegNetX-6.4GF-FPN](./mask_rcnn_regnetx_6GF_fpn_1x_coco.py)| pytorch |   1x    | ||||   |
| X-101-64x4d-FPN | pytorch |   1x    | 10.7     | 6.9            | 42.8   | 38.4    |  [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201_124310.log.json) |
|[RegNetX-8.0GF-FPN](./mask_rcnn_regnetx_8GF_fpn_1x_coco.py)| pytorch |   1x    | ||||   |

### Faster R-CNN
|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50-FPN*     |  caffe  |   1x    |          |                | 38.8   |   |
|[RegNetX-3.2GF-FPN*](./faster_rcnn_regnetx_3GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    | ||39.5|   |

### RetinaNet
|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50-FPN*     |  caffe  |   1x    |          |                | 38.8   |   |
|[RegNetX-3.2GF-FPN*](./retinanet_regnetx_3GF_fpn_mstrain_1x_coco.py)| pytorch |   1x    | ||38|   |

**Note**
1. Model marked with * behind it means this model is trained with multi-scale training.

# EfficientDet: Scalable and Efficient Object Detection

## Introduction
EfficientDet family of models based on [EfficientDet paper](arxiv.org/pdf/1911.09070.pdf).

Main features of EfficientDet models are EfficientNet backbone and BiFPN neck.

## Results and Models
|     Backbone    |    Head    |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :--------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
| efficientnet_b0 | retinahead | pytorch |   1x    |   22.0   | 33.3           |  30.2  | [model](https://storage.openvinotoolkit.org/repositories/mmdetection/models/efficientdet/retinanet_effd0_bifpn_1x_coco/epoch_300.pth) |

**Note:**

- retinahead_effd0_bifpn_1x_coco is not an official model in the EfficientDet paper. Everything but the head is similar to the model in the paper.

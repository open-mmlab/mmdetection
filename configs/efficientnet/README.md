# EfficientNet

> [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v5)

<!-- [BACKBONE] -->

## Introduction

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

## Results and Models

### RetinaNet

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                             Config                                                              |                                                                                                                                                                              Download                                                                                                                                                                              |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Efficientnet-b3 | pytorch |   1x    |    -     |       -        |  40.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco/retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806.log.json) |

## Citation

```latex
@article{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

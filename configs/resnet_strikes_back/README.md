# ResNet strikes back

> [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476)

<!-- [OTHERS] -->

## Abstract

The influential Residual Networks designed by He et al. remain the gold-standard architecture in numerous scientific publications. They typically serve as the default architecture in studies, or as baselines when new architectures are proposed. Yet there has been significant progress on best practices for training neural networks since the inception of the ResNet architecture in 2015. Novel optimization & dataaugmentation have increased the effectiveness of the training recipes.

In this paper, we re-evaluate the performance of the vanilla ResNet-50 when trained with a procedure that integrates such advances. We share competitive training settings and pre-trained models in the timm open-source library, with the hope that they will serve as better baselines for future work. For instance, with our more demanding training setting, a vanilla ResNet-50 reaches 80.4% top-1 accuracy at resolution 224×224 on ImageNet-val without extra data or distillation. We also report the performance achieved with popular models with our training procedure.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/149324625-4546a5a7-704f-406c-982f-0376a20d03d8.png"/>
</div>

## Results and Models

|       Method       | Backbone | Lr schd | Mem (GB) | Inf time (fps) |   box AP    |   mask AP   |                          Config                          |                                                                                                                                                                                                   Download                                                                                                                                                                                                   |
| :----------------: | :------: | :-----: | :------: | :------------: | :---------: | :---------: | :------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Faster R-CNN    | R-50 rsb |   1x    |   3.9    |       -        | 40.8 (+3.4) |      -      |    [Config](./faster-rcnn_r50-rsb-pre_fpn_1x_coco.py)    |             [model](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_162229-32ae82a9.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco/faster_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_162229.log.json)             |
|     Mask R-CNN     | R-50 rsb |   1x    |   4.5    |       -        | 41.2 (+3.0) | 38.2 (+3.0) |     [Config](./mask-rcnn_r50-rsb-pre_fpn_1x_coco.py)     |                 [model](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_174054-06ce8ba0.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco/mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_174054.log.json)                 |
| Cascade Mask R-CNN | R-50 rsb |   1x    |   6.2    |       -        | 44.8 (+3.6) | 39.9 (+3.6) | [Config](./cascade-mask-rcnn_r50-rsb-pre_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_193636-8b9ad50f.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco_20220113_193636.log.json) |
|     RetinaNet      | R-50 rsb |   1x    |   3.8    |       -        | 39.0 (+2.5) |      -      |     [Config](./retinanet_r50-rsb-pre_fpn_1x_coco.py)     |                 [model](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/retinanet_r50_fpn_rsb-pretrain_1x_coco/retinanet_r50_fpn_rsb-pretrain_1x_coco_20220113_175432-bd24aae9.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back/retinanet_r50_fpn_rsb-pretrain_1x_coco/retinanet_r50_fpn_rsb-pretrain_1x_coco_20220113_175432.log.json)                 |

**Notes:**

- 'rsb' is short for 'resnet strikes back'
- We have done some grid searches on learning rate and weight decay and get these optimal hyper-parameters.

## Citation

```latex
@article{wightman2021resnet,
title={Resnet strikes back: An improved training procedure in timm},
author={Ross Wightman, Hugo Touvron, Hervé Jégou},
journal={arXiv preprint arXiv:2110.00476},
year={2021}
}
```

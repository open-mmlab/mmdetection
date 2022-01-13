# ResNet strikes back: An improved training procedure in timm

## Abstract

<!-- [ABSTRACT] -->

The influential Residual Networks designed by He et al. remain the gold-standard architecture in numerous scientific publications. They typically serve as the default architecture in studies, or as baselines when new architectures are proposed. Yet there has been significant progress on best practices for training neural networks since the inception of the ResNet architecture in 2015. Novel optimization & dataaugmentation have increased the effectiveness of the training recipes.

In this paper, we re-evaluate the performance of the vanilla ResNet-50 when trained with a procedure that integrates such advances. We share competitive training settings and pre-trained models in the timm open-source library, with the hope that they will serve as better baselines for future work. For instance, with our more demanding training setting, a vanilla ResNet-50 reaches 80.4% top-1 accuracy at resolution 224×224 on ImageNet-val without extra data or distillation. We also report the performance achieved with popular models with our training procedure.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/149324625-4546a5a7-704f-406c-982f-0376a20d03d8.png"/>
</div>

<!-- [PAPER_TITLE: ResNet strikes back: An improved training procedure in timm] -->
<!-- [PAPER_URL: https://arxiv.org/pdf/2110.00476] -->

## Citation

<!-- [BACKBONE] -->

```latex
@article{wightman2021resnet,
title={Resnet strikes back: An improved training procedure in timm},
author={Ross Wightman, Hugo Touvron, Hervé Jégou},
journal={arXiv preprint arXiv:2110.00476},
year={2021}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config   | Download   |
| :-------------: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|   R-50-FPN      |    1x   |          |   -            |  40.8  | [Config](./faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py)| |

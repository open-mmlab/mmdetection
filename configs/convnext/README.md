# ConvNeXt

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

## Introduction

<!-- [BACKBONE] -->

<a href="https://github.com/facebookresearch/ConvNeXt">Official Repo</a>

<a href="https://github.com/open-mmlab/mmclassification/blob/v0.22.0/mmcls/models/backbones/convnext.py#L133">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width="90%"/>
</div>

```bibtex
@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

### Usage

- ConvNeXt backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.

```shell
pip install mmcls>=0.22.0
```

## Results and models

| Method | Backbone | Pretrain    | Lr schd | Multi-scale crop     |   FP16   |Mem (GB) | Inf time (fps) | box AP | mask AP |  Config  |   Download  |
| :------: | :------: | :---------: | :-----: | :-------------------:| :------: |:------: | :------------: | :----: | :-----: | :------: |  :--------: |
|  Mask R-CNN | ConvNeXt-T  | ImageNet-1K |    3x   |        yes            |    yes    |   xxx   |                |  46.2  |  41.7   | [config](./mask_rcnn_convnext-t_p4_w7_fpn_fp16_ms-crop_3x_coco.py)             | [model](https://download.openmmlab.com/mmdetection/v2.0/xxx)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/xxx) |
|  Cascade Mask R-CNN | ConvNeXt-T  | ImageNet-1K |    3x   |        yes           |    yes    |   xxx  |                |  50.4  |  43.7   | [config](./cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py)     | [model](https://download.openmmlab.com/mmdetection/v2.0/xxx)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/xxx) |
|  Cascade Mask R-CNN | ConvNeXt-S  | ImageNet-1K |    3x   |        yes           |    yes   |   xxx  |                |  51.9  |  45.0   | [config](./cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py)| [model](https://download.openmmlab.com/mmdetection/v2.0/xxx)  &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/xxx) |

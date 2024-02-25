# ViT-Adapter

> [Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534)

<!-- [ALGORITHM] -->

## Abstract

This work investigates a simple yet powerful dense prediction task adapter for Vision Transformer (ViT). Unlike recently advanced variants that incorporate vision-specific inductive biases into their architectures, the plain ViT suffers inferior performance on dense predictions due to weak prior assumptions. To address this issue, we propose the ViT-Adapter, which allows plain ViT to achieve comparable performance to vision-specific transformers. Specifically, the backbone in our framework is a plain ViT that can learn powerful representations from large-scale multi-modal data. When transferring to downstream tasks, a pre-training-free adapter is used to introduce the image-related inductive biases into the model, making it suitable for these tasks. We verify ViT-Adapter on multiple dense prediction tasks, including object detection, instance segmentation, and semantic segmentation. Notably, without using extra detection data, our ViT-Adapter-L yields state-of-the-art 60.9 box AP and 53.0 mask AP on COCO test-dev. We hope that the ViT-Adapter could serve as an alternative for vision-specific transformers and facilitate future research.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/202935878-97c2b136-504b-48a3-964b-b79c2b5dbff8.png"/>
</div>

## Results and Models

| Backbone | Lr schd | Mem (GB) | box AP | mask AP |                             Config                             |         Download         |
| :------: | :-----: | :------: | :----: | :-----: | :------------------------------------------------------------: | :----------------------: |
|  DeiT-T  |   3x    |          |  45.6  |  40.9   | [config](./configs/mask-rcnn_vitadapter-deit-t_fpn_3x_coco.py) | [model](<>) \| [log](<>) |
|  BEiT-B  |   3x    |          |  48.7  |  43.1   |   [config](./configs/mask-rcnn_beitadapter-b_fpn_3x_coco.py)   | [model](<>) \| [log](<>) |

## Usage

### Install additional requirements

RepVGG backbone needs to install [MMClassification](https://github.com/open-mmlab/mmclassification) first, which has abundant backbones for downstream tasks.
If you have already installed requirements for mmdet, run

```shell
pip install 'mmcls>=1.0'
```

See [this document](https://mmclassification.readthedocs.io/en/latest/install.html) for the details of MMClassification installation.

## Citation

```latex
@article{chen2022vitadapter,
  title={Vision Transformer Adapter for Dense Predictions},
  author={Chen, Zhe and Duan, Yuchen and Wang, Wenhai and He, Junjun and Lu, Tong and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.08534},
  year={2022}
}
```

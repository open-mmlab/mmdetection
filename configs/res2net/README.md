# Res2Net: A New Multi-scale Backbone Architecture

## Abstract

<!-- [ABSTRACT] -->

Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143972411-8d08113f-9fce-4d24-a138-4fadf2c54f9a.png" height="300"/>
</div>

<!-- [PAPER_TITLE: Res2Net: A New Multi-scale Backbone Architecture] -->
<!-- [PAPER_URL: https://arxiv.org/abs/1904.01169] -->

## Introduction

<!-- [ALGORITHM] -->

We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

|    Backbone     |Params. | GFLOPs  | top-1 err. | top-5 err. |
| :-------------: |:----:  | :-----: | :--------: | :--------: |
| ResNet-101      |44.6 M  | 7.8     |  22.63     |  6.44      |
| ResNeXt-101-64x4d |83.5M | 15.5    |  20.40     |  -         |
| HRNetV2p-W48    | 77.5M  | 16.1    |  20.70     |  5.50      |
| Res2Net-101     | 45.2M  | 8.3     |  18.77     |  4.64      |

Compared with other backbone networks, Res2Net requires fewer parameters and FLOPs.

**Note:**

- GFLOPs for classification are calculated with image size (224x224).

## Citation

```latex
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758},
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|R2-101-FPN       | pytorch |   2x   |   7.4    |   -           |  43.0  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco_20200514_231734.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|R2-101-FPN       | pytorch |    2x   |   7.9    |      -         |   43.6 | 38.7  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/mask_rcnn_r2_101_fpn_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/res2net/mask_rcnn_r2_101_fpn_2x_coco/mask_rcnn_r2_101_fpn_2x_coco-17f061e8.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/res2net/mask_rcnn_r2_101_fpn_2x_coco/mask_rcnn_r2_101_fpn_2x_coco_20200515_002413.log.json) |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|R2-101-FPN       | pytorch |   20e   |   7.8    |      -         |  45.7  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco_20200515_091644.log.json) |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
R2-101-FPN       | pytorch |  20e   |    9.5  |      -         |  46.4  |  40.0  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco_20200515_091645.log.json) |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
| R2-101-FPN     | pytorch |   20e   |    -    |      -         |  47.5  | 41.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/htc_r2_101_fpn_20e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/res2net/htc_r2_101_fpn_20e_coco/htc_r2_101_fpn_20e_coco-3a8d2112.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/res2net/htc_r2_101_fpn_20e_coco/htc_r2_101_fpn_20e_coco_20200515_150029.log.json) |

- Res2Net ImageNet pretrained models are in [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels).
- More applications of Res2Net are in [Res2Net-Github](https://github.com/Res2Net/).

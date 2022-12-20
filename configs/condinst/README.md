# CondInst

> [CondInst: Conditional Convolutions for Instance
> Segmentation](https://arxiv.org/pdf/2003.05664.pdf)

<!-- [ALGORITHM] -->

## Abstract

We propose a simple yet effective instance segmentation framework, termed CondInst (conditional convolutions for instance segmentation). Top-performing instance segmentation methods such as Mask
R-CNN rely on ROI operations (typically ROIPool or ROIAlign) to
obtain the final instance masks. In contrast, we propose to solve instance segmentation from a new perspective. Instead of using instancewise ROIs as inputs to a network of fixed weights, we employ dynamic
instance-aware networks, conditioned on instances. CondInst enjoys two
advantages: 1) Instance segmentation is solved by a fully convolutional
network, eliminating the need for ROI cropping and feature alignment.
2\) Due to the much improved capacity of dynamically-generated conditional convolutions, the mask head can be very compact (e.g., 3 conv.
layers, each having only 8 channels), leading to significantly faster inference. We demonstrate a simpler instance segmentation method that can
achieve improved performance in both accuracy and inference speed. On
the COCO dataset, we outperform a few recent methods including welltuned Mask R-CNN baselines, without longer training schedules needed.

<div align=center>
<img src="https://user-images.githubusercontent.com/57584090/203303488-3dbc36da-09a6-4dc8-be9d-d9af27bd1234.png"/>
</div>

## Results and Models

| Backbone |  Style  | MS train | Lr schd | bbox AP | mask AP |                          Config                           |                                                                                                                                                                              Download                                                                                                                                                                              |
| :------: | :-----: | :------: | :-----: | :-----: | :-----: | :-------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |    Y     |   1x    |  39.8   |  36.0   | [config](./condinst_r50_fpn_ms-poly-90k_coco_instance.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223.json) |

## Citation

```latex
@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

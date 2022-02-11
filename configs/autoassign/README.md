# AutoAssign

> [AutoAssign: Differentiable Label Assignment for Dense Object Detection](https://arxiv.org/abs/2007.03496)

<!-- [ALGORITHM] -->

## Abstract

Determining positive/negative samples for object detection is known as label assignment. Here we present an anchor-free detector named AutoAssign. It requires little human knowledge and achieves appearance-aware through a fully differentiable weighting mechanism. During training, to both satisfy the prior distribution of data and adapt to category characteristics, we present Center Weighting to adjust the category-specific prior distributions. To adapt to object appearances, Confidence Weighting is proposed to adjust the specific assign strategy of each instance. The two weighting modules are then combined to generate positive and negative weights to adjust each location's confidence. Extensive experiments on the MS COCO show that our method steadily surpasses other best sampling strategies by large margins with various backbones. Moreover, our best model achieves 52.1% AP, outperforming all existing one-stage detectors. Besides, experiments on other datasets, e.g., PASCAL VOC, Objects365, and WiderFace, demonstrate the broad applicability of AutoAssign.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143870875-33567e44-0584-4470-9a90-0df0fb6c1fe2.png"/>
</div>

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50     | caffe | 1x      | 4.08      |   40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.3 mAP. mAP 40.3 ~ 40.6 is acceptable. Such fluctuation can also be found in the original implementation.
2. You can get a more stable results ~ mAP 40.6 with a schedule total 13 epoch, and learning rate is divided by 10 at 10th and 13th epoch.

## Citation

```latex
@article{zhu2020autoassign,
  title={AutoAssign: Differentiable Label Assignment for Dense Object Detection},
  author={Zhu, Benjin and Wang, Jianfeng and Jiang, Zhengkai and Zong, Fuhang and Liu, Songtao and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2007.03496},
  year={2020}
}
```

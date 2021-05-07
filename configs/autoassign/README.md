# AutoAssign: Differentiable Label Assignment for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

```
@article{zhu2020autoassign,
  title={AutoAssign: Differentiable Label Assignment for Dense Object Detection},
  author={Zhu, Benjin and Wang, Jianfeng and Jiang, Zhengkai and Zong, Fuhang and Liu, Songtao and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2007.03496},
  year={2020}
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50     | pytorch | 1x      | 4.08      |   40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.3 mAP. mAP 40.3 ~ 40.6 is acceptable. Such fluctuation can also be found in the original implementation.
2. You can get a more stable results ~ mAP 40.6 with a schedule total 13 epoch, and learning rate is divided by 10 at 10th and 13th epoch.

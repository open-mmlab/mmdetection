# Group Normalization

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{wu2018group,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## Results and Models

| Backbone      | model      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.1      | 11.0           | 40.2   | 36.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206_050355.log.json) |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.1      | -              | 40.5   | 36.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214_063512.log.json) |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 9.0            | 41.9   | 37.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205_234402.log.json) |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      |                | 42.1   | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.1      | 10.9           | 40.0   | 36.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207_225832.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.1      | -              | 40.1   | 36.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225_235135.log.json) |

**Notes:**

- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- **Memory, Train/Inf time is outdated.**

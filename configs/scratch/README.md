# Rethinking ImageNet Pre-training

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{he2018rethinking,
  title={Rethinking imagenet pre-training},
  author={He, Kaiming and Girshick, Ross and Doll{\'a}r, Piotr},
  journal={arXiv preprint arXiv:1811.08883},
  year={2018}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | box AP | mask AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN | R-50-FPN  | pytorch | 6x      | 40.7   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_20200201_193013.log.json) |
| Mask R-CNN   | R-50-FPN  | pytorch | 6x      | 41.2   | 37.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_20200201_193051.log.json)  |

Note:

- The above models are trained with 16 GPUs.

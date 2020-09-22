# FreeAnchor: Learning to Match Anchors for Visual Object Detection

## Introduction

```
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```

## Results and Models

| Backbone | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:--------:|:-------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| R-50     | pytorch | 1x      | 4.9      | 18.4 | 38.7 | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130_095625.log.json) |
| R-101       | pytorch | 1x   | 6.8      | 14.9 | 40.3 | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130_100723.log.json) |
| X-101-32x4d | pytorch | 1x   | 8.1      | 11.1 | 41.9 | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130_095627.log.json) |

**Notes:**
- We use 8 GPUs with 2 images/GPU.
- For more settings and models, please refer to the [official repo](https://github.com/zhangxiaosong18/FreeAnchor).

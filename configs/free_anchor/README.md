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

| Backbone    | Style   | Lr schd | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------------------:|:--------------:|:------:|:--------:|
| R-50        | pytorch | 1x      | 4.7 | 0.322 | 12.0 | 38.4 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/free_anchor/retinanet_free_anchor_r50_fpn_1x_20190914-84db6585.pth) |
| R-101       | pytorch | 1x      | 6.6 | 0.437 | 9.7 | 40.3 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/free_anchor/retinanet_free_anchor_r101_fpn_1x_20190914-c4e4db81.pth) |
| X-101-32x4d | pytorch | 1x      | 7.8 | 0.640 | 8.4 | 42.0 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/free_anchor/retinanet_free_anchor_x101-32x4d_fpn_1x_20190914-eb73b804.pth) |

**Notes:**
- We use 8 GPUs with 2 images/GPU.
- For more settings and models, please refer to the [official repo](https://github.com/zhangxiaosong18/FreeAnchor).

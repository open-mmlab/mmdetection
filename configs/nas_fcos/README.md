# NAS-FCOS: Fast Neural Architecture Search for Object Detection

## Introduction

```
@article{wang2019fcos,
  title={Nas-fcos: Fast neural architecture search for object detection},
  author={Wang, Ning and Gao, Yang and Chen, Hao and Wang, Peng and Tian, Zhi and Shen, Chunhua},
  journal={arXiv preprint arXiv:1906.04423},
  year={2019}
}
```

## Results and Models

| Head      | Backbone  | Style   | GN      | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:---------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:--------:|
| NAS-FCOSHead | R-50   | caffe   | Y       | 1x      | 8.53     | 8.73           | 39.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn_4x4_1x_coco_bbox_mAP-0.394_20200520_151831-1bdba3ce.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn_4x4_1x_coco_20200520_151831.log.json) |
| FCOSHead  | R-50      | caffe   | Y       | 1x      | 8.39     | 9.68           | 38.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn_4x4_1x_coco_bbox_mAP-0.385_20200521_103823-7fdcbce0.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn_4x4_1x_coco_20200521_103823.log.json) |

**Notes:**
- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU.

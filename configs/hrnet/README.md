# High-resolution networks (HRNets) for object detection

## Introduction

```
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao 
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Results and Models

Faster R-CNN

| Backbone|#Params|GFLOPs|Lr sched|mAP|Download|
| :--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 |26.2M|159.1| 1x | 36.1 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w18_fpn_1x_20190522-e368c387.pth)|
| HRNetV2-W18 |26.2M|159.1| 20-23-24e | 38.1 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w18_fpn_20_23_24e_20190522-ed3c0293.pth)|
| HRNetV2-W32 |45.0M|245.3| 1x | 39.5 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w32_fpn_1x_20190522-d22f1fef.pth)|
| HRNetV2-W32 |45.0M|245.3| 20-23-24e | 40.8 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w32_fpn_20_23_24e_20190522-2d67a5eb.pth)|
| HRNetV2-W40 |60.5M|314.9| 1x | 40.4 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w40_fpn_1x_20190522-30502318.pth)|
| HRNetV2-W40 |60.5M|314.9| 20-23-24e | 41.4 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w40_fpn_20_23_24e_20190522-050a7c7f.pth)|


Mask R-CNN

|Backbone|Lr sched|mask mAP|box mAP|Download|
|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 | 1x | 34.2 | 37.3 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w18_fpn_1x_20190522-c8ad459f.pth)|
| HRNetV2-W18 | 20-23-24e | 35.7 | 39.2 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w18_fpn_20_23_24e_20190522-5c11b7f2.pth)|
| HRNetV2-W32 | 1x | 36.8 | 40.7 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w32_fpn_1x_20190522-374aaa00.pth)|
| HRNetV2-W32 | 20-23-24e | 37.6 | 42.1 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w32_fpn_20_23_24e_20190522-4dd02a79.pth)|

Cascade R-CNN

|Backbone|Lr sched|mAP|Download|
|:--:|:--:|:--:|:--:|
| HRNetV2-W32 | 20e | 43.7 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth)|

**Note:**

- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

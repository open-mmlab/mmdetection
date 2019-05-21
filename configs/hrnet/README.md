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
| HRNetV2-W18 |26.2M|159.1| 1x | 36.1 | [model](https://1drv.ms/u/s!Avk3cZ0cr1Jeake7zkUDn0zJE8w)|
| HRNetV2-W18 |26.2M|159.1| 2x | 38.1 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JeaeLG5k4Ic7WzHHc)|
| HRNetV2-W32 |45.0M|245.3| 1x | 39.5 | [model](https://1drv.ms/u/s!Avk3cZ0cr1Jea028vq-D4fP8Hac)|
| HRNetV2-W32 |45.0M|245.3| 2x | 40.8 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JebPbIUQndxFZGf84)|
| HRNetV2-W40 |60.5M|314.9| 1x | 40.4 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JebW7qRwQSWGHhaXw)|
| HRNetV2-W40 |60.5M|314.9| 2x | 41.4 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JebwOjVjwjUxzAWso)|


Mask R-CNN

|Backbone|Lr sched|mask mAP|box mAP|Download|
|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 | 1x | 34.2 | 37.3 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JebvCt1X1LUiWEMs8)|
| HRNetV2-W18 | 2x | 35.7 | 39.2 | [model](https://1drv.ms/u/s!Avk3cZ0cr1Jec5coSg_yGhQVJUk)|
| HRNetV2-W32 | 1x | 36.8 | 40.7 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JedCa_MG1Qage0JkE)|
| HRNetV2-W32 | 2x | 37.6 | 42.1 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JedUGjiRI3Ld3Vbqs)|

Cascade R-CNN

|Backbone|Lr sched|mAP|Download|
|:--:|:--:|:--:|:--:|
| HRNetV2-W32 | 20e | 43.7 | [model](https://1drv.ms/u/s!Avk3cZ0cr1JedmgHhUA7kIgloTc)|
| R-101-FPN   | 20e | 42.6 | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth) |

**Note:**

- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

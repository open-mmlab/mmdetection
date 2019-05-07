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

| Backbone|#Params|GFLOPs|lr sched|mAP|pretrained model|detection model|
| :--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 |26.2M|159.1| 1x | 36.1 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzaTqcKb9QJrIZS7Y)|
| HRNetV2-W18 |26.2M|159.1| 2x | 38.1 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [FasterR-CNN-HR18-2x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzcHt7xyWTgVxmMLw)|
| HRNetV2-W32 |45.0M|245.3| 1x | 39.5 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzaxRamJewuDqSozQ)|
| HRNetV2-W32 |45.0M|245.3| 2x | 40.8 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [FasterR-CNN-HR32-2x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbE6rbdU9whYJkqs)|
| HRNetV2-W40 |60.5M|314.9| 1x | 40.4 | [HRNetV2-W40](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo) | [FasterR-CNN-HR40-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbE6rbdU9whYJkqs)|
| HRNetV2-W40 |60.5M|314.9| 2x | 41.4 | [HRNetV2-W40](https://1drv.ms/u/s!Aus8VCZ_C_33ck0gvo5jfoWBOPo) | [FasterR-CNN-HR40-2x.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzb1Uy6QLZnsyfuFc)|


Mask R-CNN

|Backbone|lr sched|Mask mAP|Box mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNetV2-W18 | 1x | 34.2 | 37.3 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [MaskR-CNN-HR18-1x.pth](https://1drv.ms/u/s!AiWjZ1Lamlxzcfh06SXd2GR1zKw)|
| HRNetV2-W18 | 2x | 35.7 | 39.2 | [HRNetV2-W18](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw) | [MaskR-CNN-HR18-2x.pth](https://1drv.ms/u/s!AjfnYvdHLH5TafSZNlgq6UWnJWk)|
| HRNetV2-W32 | 1x | 36.8 | 40.7 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [MaskR-CNN-HR32-1x.pth](https://1drv.ms/u/s!AiWjZ1LamlxzcugO3KlXfy_YhiE)|
| HRNetV2-W32 | 2x | 37.6 | 42.1 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [MaskR-CNN-HR32-2x.pth](https://1drv.ms/u/s!AjfnYvdHLH5Taqt21comOmTbdBg)|

Cascade R-CNN

|Backbone|lr sched|mAP|pretrained model|detection model|
|:--:|:--:|:--:|:--:|:--:|
| ResNet-101  | 20e | 42.8 | [ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [CascadeR-CNN-R101-20e.pth](https://1drv.ms/u/s!AiWjZ1LamlxzbvOFlCnGhXhKmsY)|
| HRNetV2-W32 | 20e | 43.7 | [HRNetV2-W32](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w) | [CascadeR-CNN-HR32-20e.pth](https://1drv.ms/u/s!AiWjZ1LamlxzasFUt8GWHW1Og3I)|

**Note:**

- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

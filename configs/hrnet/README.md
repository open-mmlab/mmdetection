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


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2-W18   | pytorch |   1x    |  36.1  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w18_fpn_1x_20190522-e368c387.pth) |
|   HRNetV2-W18   | pytorch |   2x    |  38.3  | [model](https://1drv.ms/u/s!Auszw0Nn8DCRdjF_JWGaZJbUNr4?e=sqxJHZ) |
|   HRNetV2-W32   | pytorch |   1x    |  39.5  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w32_fpn_1x_20190522-d22f1fef.pth) |
|   HRNetV2-W32   | pytorch |   2x    |  40.6  | [mdoel](https://1drv.ms/u/s!Auszw0Nn8DCReTjC__GBpeHuIRU?e=ZNFXrP) |


### Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   1x    |  37.3  |  34.2   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w18_fpn_1x_20190522-c8ad459f.pth) |
|   HRNetV2-W18   | pytorch |   2x    |  39.2  |  35.7   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wdrfShb-tRYEXFN4?e=vKBNQk) |
|   HRNetV2-W32   | pytorch |   1x    |  40.7  |  36.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w32_fpn_1x_20190522-374aaa00.pth) |
|   HRNetV2-W32   | pytorch |   2x    |  41.7  |  37.5  | [model](https://1drv.ms/u/s!AiO3BIXgPy5wd6Ls0x644doGTRI?e=GMJk6i)  |


### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  41.2  | [model](https://1drv.ms/u/s!Auszw0Nn8DCRdYrajEKsQ3NGDJc?e=TKXiHz) |
|   HRNetV2-W32   | pytorch |   20e   |  43.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth)|
|   HRNetV2-W48   | pytorch |   20e   |  44.6  | [model](https://1drv.ms/u/s!Auszw0Nn8DCReL0DDCoam7I49bg?e=I5qMFf) |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  41.9  |  36.4   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRcqFXodE8XZyW_L4?e=QV8FSF) |
|   HRNetV2-W32   | pytorch |   20e   |  44.5  |  38.5   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRdA6l34DNpT7sTQM?e=ZvCgjV) |
|   HRNetV2-W48   | pytorch |   20e   |  46.0  |  39.5   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRd2E7qOkocmi27Vk?e=acVJNl) |


### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  43.1  |  37.9   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRepy7NYQM8BpvLBM?e=fZucBh) |
|   HRNetV2-W32   | pytorch |   20e   |  45.3  |  39.6   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRezdRHZQ38kIgaQE?e=N7SJnK) |
|   HRNetV2-W48   | pytorch |   20e   |  46.8  |  40.7   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRfRkTT1_9Dtug3OM?e=nFPPFR) |
|   HRNetV2-W48   | pytorch |   28e   |  47.0  |  41.0   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRfEr2jc5WbTI3R68?e=T4vN10) |
| X-101-64x4d-FPN | pytorch |   28e   |  46.8  |  40.7   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRfteGxLh_miS0efw?e=sDMXUS) |


### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:--------:|
|HRNetV2-W18| pytorch | Y       | N       | 1x       | 35.2   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wcXPTJq-_nNp2V7M?e=NEzyJp) |
|HRNetV2-W18| pytorch | Y       | N       | 2x       | 38.2   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wcv55LdxPqcuRHj0?e=olSrQh) |
|HRNetV2-W32| pytorch | Y       | N       | 1x       | 37.7   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wc8N7SaMZkLA311w?e=Fg98sq) |
|HRNetV2-W32| pytorch | Y       | N       | 2x       | 40.3   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wdFnsCU3Q-iv5MtI?e=gh55wL) |
|HRNetV2-W18| pytorch | Y       | Y       | 2x       | 38.1   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wcKVw15IpwGIriZk?e=Ge2KiM) |
|HRNetV2-W32| pytorch | Y       | Y       | 2x       | 41.4   | [model](https://1drv.ms/u/s!AiO3BIXgPy5wdY0bAy_pESzgSeA?e=S3qMLC) |
|HRNetV2-W48| pytorch | Y       | Y       | 2x       | 42.9   | [model](https://1drv.ms/u/s!AiO3BIXgPy5weGKcdvGgQXoRPdM?e=1q555d) |



**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

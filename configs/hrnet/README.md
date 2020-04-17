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
|   HRNetV2p-W18   | pytorch |   1x    |       | [model]() &#124; [log]() |
|   HRNetV2p-W18   | pytorch |   2x    |       | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   1x    |       | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   2x    |       | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   1x    |       | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   2x    |       | [model]() &#124; [log]() |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   1x    |       |        | [model]() &#124; [log]() |
|   HRNetV2p-W18   | pytorch |   2x    |       |        | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   1x    |       |        | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   2x    |       |        | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   1x    |       |        | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   2x    |       |        | [model]() &#124; [log]() |


### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |       | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   20e   |       | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   20e   |       | [model]() &#124; [log]() |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |       |         | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   20e   |       |         | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   20e   |       |         | [model]() &#124; [log]() |


### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |       |         | [model]() &#124; [log]() |
|   HRNetV2p-W32   | pytorch |   20e   |       |         | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   20e   |       |         | [model]() &#124; [log]() |
|   HRNetV2p-W48   | pytorch |   28e   |       |         | [model]() &#124; [log]() |
| X-101-64x4d-FPN  | pytorch |   28e   |       |         | [model]() &#124; [log]() |


### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 13.0 | 9.9 | 35.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20200316-c24bac34.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20200316_103815.log.json) |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 13.0 | 9.7 | 37.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20200316-15348c5b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20200316_103815.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 17.5 | 9.5 | 39.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20200314-59a7807f.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20200314_150555.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 17.5 | 9.8 | 40.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20200314-faf8f0b8.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20200314_145136.log.json) |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 13.0 | 9.7 | 38.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20200316-a668468b.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20200316_104027.log.json) |
|HRNetV2p-W32| pytorch | Y       | Y       | 2x       | 17.5 | 9.1 | 41.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20200314-065d37a6.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20200314_145356.log.json) |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 20.3 | 9.9 | 42.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20200314-e201886d.pth) &#124; [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20200314_150607.log.json) |



**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

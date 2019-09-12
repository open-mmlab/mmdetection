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
|   HRNetV2p-W18   | pytorch |   1x    |  36.1  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w18_1x_20190522-e368c387.pth) |
|   HRNetV2p-W18   | pytorch |   2x    |  38.3  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w18_2x_20190810-9c8615d5.pth) |
|   HRNetV2p-W32   | pytorch |   1x    |  39.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth) |
|   HRNetV2p-W32   | pytorch |   2x    |  40.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_2x_20190810-24e8912a.pth) |
|   HRNetV2p-W48   | pytorch |   1x    |  40.9  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w48_1x_20190820-5c6d0903.pth)    |
|   HRNetV2p-W48   | pytorch |   2x    |  41.5  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w48_2x_20190820-79fb8bfc.pth) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   1x    |  37.3  |  34.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w18_1x_20190522-c8ad459f.pth) |
|   HRNetV2p-W18   | pytorch |   2x    |  39.2  |  35.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w18_2x_20190810-1e4747eb.pth) |
|   HRNetV2p-W32   | pytorch |   1x    |  40.7  |  36.8   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w32_1x_20190522-374aaa00.pth) |
|   HRNetV2p-W32   | pytorch |   2x    |  41.7  |  37.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w32_2x_20190810-773eca75.pth) |
|   HRNetV2p-W48   | pytorch |   1x    |  42.4  |  38.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w48_1x_20190820-0923d1ad.pth) |
|   HRNetV2p-W48   | pytorch |   2x    |  42.9  |  38.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w48_2x_20190820-70df51b2.pth) |


### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |  41.2  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_rcnn_hrnetv2p_w18_20e_20190810-132012d0.pth)|
|   HRNetV2p-W32   | pytorch |   20e   |  43.7  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth)|
|   HRNetV2p-W48   | pytorch |   20e   |  44.6  | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_rcnn_hrnetv2p_w48_20e_20190810-f40ed8e1.pth)|


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |  41.9  |  36.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_20190810-054fb7bf.pth) |
|   HRNetV2p-W32   | pytorch |   20e   |  44.5  |  38.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth) |
|   HRNetV2p-W48   | pytorch |   20e   |  46.0  |  39.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_mask_rcnn_hrnetv2p_w48_20e_20190810-d04a1415.pth) |


### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2p-W18   | pytorch |   20e   |  43.1  |  37.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_hrnetv2p_w18_20e_20190810-d70072af.pth) |
|   HRNetV2p-W32   | pytorch |   20e   |  45.3  |  39.6   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_hrnetv2p_w32_20e_20190810-82f9ef5a.pth) |
|   HRNetV2p-W48   | pytorch |   20e   |  46.8  |  40.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_hrnetv2p_w48_20e_20190810-f6d2c3fd.pth) |
|   HRNetV2p-W48   | pytorch |   28e   |  47.0  |  41.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_hrnetv2p_w48_28e_20190810-a4274b38.pth) |
| X-101-64x4d-FPN | pytorch |   28e   |  46.8  |  40.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_x101_64x4d_28e_20190810-d7c19dc0.pth) |


### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | box AP | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 35.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w18_1x_20190810-87a17998.pth) |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 38.2   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w18_2x_20190810-dfd60a7b.pth) |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 37.7   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w32_1x_20190810-62014622.pth) |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 40.3   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w32_2x_20190810-8e987ec1.pth) |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 38.1   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w18_mstrain_2x_20190810-eb846b2c.pth) |
|HRNetV2p-W32| pytorch | Y       | Y       | 2x       | 41.4   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w32_mstrain_2x_20190810-96127bf8.pth) |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 42.9   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/fcos_hrnetv2p_w48_mstrain_2x_20190810-f7dc8801.pth) |



**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

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
|   HRNetV2-W18   | pytorch |   2x    |  38.3  | [model](https://1drv.ms/u/s!Auszw0Nn8DCRbh41jpZ323XI--k?e=U4YpqN) |
|   HRNetV2-W32   | pytorch |   1x    |  39.5  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/faster_rcnn_hrnetv2_w32_fpn_1x_20190522-d22f1fef.pth) |
|   HRNetV2-W32   | pytorch |   2x    |  40.6  | [mdoel](https://1drv.ms/u/s!Auszw0Nn8DCRb79TmOa-cqUhNRU?e=SvbNm8) |


### Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   1x    |  37.3  |  34.2   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w18_fpn_1x_20190522-c8ad459f.pth) |
|   HRNetV2-W18   | pytorch |   2x    |  39.2  |  35.7   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRcIXRhhKqCTqm6lM?e=4RayXL) |
|   HRNetV2-W32   | pytorch |   1x    |  40.7  |  36.8   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/mask_rcnn_hrnetv2_w32_fpn_1x_20190522-374aaa00.pth) |
|   HRNetV2-W32   | pytorch |   2x    |  41.7  |  37.5  | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFbs0M75ceGOJq5_8?e=Da64Ik)  |


### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | box AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  41.2  | [model](https://1drv.ms/u/s!Auszw0Nn8DCRbOZFb-P5mQnROag?e=FhjtRO) |
|   HRNetV2-W32   | pytorch |   20e   |  43.7  | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth)|
|   HRNetV2-W48   | pytorch |   20e   |  44.6  | [model](https://1drv.ms/u/s!Auszw0Nn8DCRbR4_fUMQIh4HLpY?e=tg4mcC) |


### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  41.9  |  36.4   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRae1ZHlXqhq6rYRs?e=3ccezX) |
|   HRNetV2-W32   | pytorch |   20e   |  44.5  |  38.5   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRapMDzVpuCtKFco8?e=QkmmGb) |
|   HRNetV2-W48   | pytorch |   20e   |  46.0  |  39.5   | [model](https://1drv.ms/u/s!Auszw0Nn8DCRa98jErb27sfqyIM?e=R7Um29) |


### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | box AP | mask AP |       Download      |
| :-------------: | :-----: | :-----: | :----: | :----:  | :-----------------: |
|   HRNetV2-W18   | pytorch |   20e   |  43.1  |  37.9   | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFaVxu0sx7ANFxQ5Q?e=Fgbl34)   |   |
|   HRNetV2-W32   | pytorch |   20e   |  45.3  |  39.6   | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFauCD-HFOaYNKf_s?e=xCZc97)   |
|   HRNetV2-W48   | pytorch |   20e   |  46.8  |  40.7   | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFbCfbs7-84a0Ab50?e=2c1YL0)   |
| X-101-64x4d-FPN | pytorch |   28e   |  46.8  |  40.7   | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFbRy5RsHSGilbdgc?e=Pao0CF)   |
|   HRNetV2-W48   | pytorch |   28e   |  47.0  |  41.0   | [model](https://1drv.ms/u/s!Ah3Ku6_VdFEFay0EW03fWCe8hn0?e=DT9MNQ)   |



**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).

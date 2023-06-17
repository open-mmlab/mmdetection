# Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking

## Abstract

<!-- [ABSTRACT] -->

Multi-Object Tracking (MOT) has rapidly progressed with the development of object detection and re-identification. However, motion modeling, which facilitates object association by forecasting short-term trajec- tories with past observations, has been relatively under-explored in recent years. Current motion models in MOT typically assume that the object motion is linear in a small time window and needs continuous observations, so these methods are sensitive to occlusions and non-linear motion and require high frame-rate videos. In this work, we show that a simple motion model can obtain state-of-the-art tracking performance without other cues like appearance. We emphasize the role of “observation” when recovering tracks from being lost and reducing the error accumulated by linear motion models during the lost period. We thus name the proposed method as Observation-Centric SORT, OC-SORT for short. It remains simple, online, and real-time but improves robustness over occlusion and non-linear motion. It achieves 63.2 and 62.1 HOTA on MOT17 and MOT20, respectively, surpassing all published methods. It also sets new states of the art on KITTI Pedestrian Tracking and DanceTrack where the object motion is highly non-linear

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/17743251/168193097-b3ad1a94-b18c-4b14-b7b1-5f8c6ed842f0.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```

## Results and models on MOT17

The performance on `MOT17-half-val` is comparable with the performance from [the OC-SORT official implementation](https://github.com/noahcao/OC_SORT). We use the same YOLO-X detector weights as in [ByteTrack](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack).

| Method  | Detector |        Train Set        | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                          Config                           |                                                                                                                                               Download                                                                                                                                               |
| :-----: | :------: | :---------------------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :-------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| OC-SORT | YOLOX-X  | CrowdHuman + half-train | half-val |   N    |       -        | 67.5 | 77.5 | 78.2 | 15987 | 19590 |  855  | [config](ocsort_yolox_x_crowdhuman_mot17-private-half.py) | [model](https://download.openmmlab.com/mmtracking/mot/ocsort/mot_dataset/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/ocsort/mot_dataset/ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618.log.json) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

OCSORT training is same as Bytetrack, please refer to [document](../../configs/bytetrack/README.md).

### 4. Testing and evaluation

OCSORT evaluation and test are same as Bytetrack, please refer to [document](../../configs/bytetrack/README.md).

### 5.Inference

OCSORT inference is same as Bytetrack, please refer to [document](../../configs/bytetrack/README.md).

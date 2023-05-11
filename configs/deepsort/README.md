# Simple online and realtime tracking with a deep association metric

## Abstract

<!-- [ABSTRACT] -->

Simple Online and Realtime Tracking (SORT) is a pragmatic approach to multiple object tracking with a focus on simple, effective algorithms. In this paper, we integrate appearance information to improve the performance of SORT. Due to this extension we are able to track objects through longer periods of occlusions, effectively reducing the number of identity switches. In spirit of the original framework we place much of the computational complexity into an offline pre-training stage where we learn a deep association metric on a largescale person re-identification dataset. During online application, we establish measurement-to-track associations using nearest neighbor queries in visual appearance space. Experimental evaluation shows that our extensions reduce the number of identity switches by 45%, achieving overall competitive performance at high frame rates.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/26813582/145542023-22950508-b35f-41b6-bc78-33d6a82bc3c3.png"/>
</div>

## Results and models on MOT17

Currently we do not support training ReID models for DeepSORT.
We directly use the ReID model from [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw). These missed features will be supported in the future.

|  Method  |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                       Config                                       |                                                                                                         Download                                                                                                         |
| :------: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepSORT | R50-FasterRCNN-FPN | R50  | half-train | half-val |   N    |      13.8      | 57.0 | 63.7 | 69.5 | 15063 | 40323 | 3276  | [config](deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

We implement DeepSORT with independent detector and ReID models.
Note that, due to the influence of parameters such as learning rate in default configuration file,
we recommend using 8 GPUs for training in order to reproduce accuracy.

You can train the detector as follows.

```shell script
# Training Faster R-CNN on mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_train.sh configs/sort/faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 4. Testing and evaluation

### 4.1 Example on MOTxx-halfval dataset

**4.1.1 use separate trained detector and reid model to evaluating and testing**

```shell
# Example 1: Test on motXX-half-val set.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_test_tracking.sh configs/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 --detector ${DETECTOR_CHECKPOINT_PATH} --reid ${REID_CHECKPOINT_PATH}
```

**4.1.2 use video_baesd to evaluating and testing**

we also provide two_ways(img_based or video_based) to evaluating and testing.
if you want to use video_based to evaluating and testing, you can modify config as follows

```
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False))
```

### 4.2 Example on MOTxx-test dataset

If you want to get the results of the [MOT Challenge](https://motchallenge.net/) test set,
please use the following command to generate result files that can be used for submission.
It will be stored in `./mot_17_test_res`, you can modify the saved path in `test_evaluator` of the config.

```shell script
# Example 2: Test on motxx-test set
# The number after config file represents the number of GPUs used
bash tools/dist_test_tracking.sh configs/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17train_test-mot17test 8 --detector ${DETECTOR_CHECKPOINT_PATH} --reid ${REID_CHECKPOINT_PATH}
```

If you want to know about more detailed usage of `test_tracking.py/dist_test_tracking.sh/slurm_test_tracking.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 5.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17train_test-mot17test --detector ${DETECTOR_CHECKPOINT_PATH} --reid ${REID_CHECKPOINT_PATH} --out mot.mp4
```

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{wojke2017simple,
  title={Simple online and realtime tracking with a deep association metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE international conference on image processing (ICIP)},
  pages={3645--3649},
  year={2017},
  organization={IEEE}
}
```

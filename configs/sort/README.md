# Simple online and realtime tracking

## Abstract

<!-- [ABSTRACT] -->

This paper explores a pragmatic approach to multiple object tracking where the main focus is to associate objects efficiently for online and realtime applications. To this end, detection quality is identified as a key factor influencing tracking performance, where changing the detector can improve tracking by up to 18.9%. Despite only using a rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers. Furthermore, due to the simplicity of our tracking method, the tracker updates at a rate of 260 Hz which is over 20x faster than other state-of-the-art trackers.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/99722489/176848133-d6621813-7b8f-4b25-96cd-2fbcc87983ce.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{bewley2016simple,
  title={Simple online and realtime tracking},
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={3464--3468},
  year={2016},
  organization={IEEE}
}
```

## Results and models on MOT17

| Method |      Detector      | ReID | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                     Config                                     |                                                       Download                                                       |
| :----: | :----------------: | :--: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :----------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|  SORT  | R50-FasterRCNN-FPN |  -   | half-train | half-val |   N    |      18.6      | 52.0 | 62.0 | 57.8 | 15150 | 40410 | 5847  | [config](sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

We implement SORT with independent detector models.
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

**4.1.1 use separate trained detector model to evaluating and testing**\*

```shell script
# Example 1: Test on motXX-half-val set.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_test_tracking.sh configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 --detector ${DETECTOR_CHECKPOINT_PATH}
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
bash tools/dist_test_tracking.sh configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17train_test-mot17test.py 8 --detector ${DETECTOR_CHECKPOINT_PATH}
```

If you want to know about more detailed usage of `test_tracking.py/dist_test_tracking.sh/slurm_test_tracking.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 5.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py --detector ${DETECTOR_CHECKPOINT_PATH}  --out mot.mp4
```

If you want to know about more detailed usage of `mot_demo.py`, please refer to this [document](../../docs/en/user_guides/tracking_inference.md).

# ByteTrack: Multi-Object Tracking by Associating Every Detection Box

## Abstract

<!-- [ABSTRACT] -->

Multi-object tracking (MOT) aims at estimating bounding boxes and identities of objects in videos. Most methods obtain identities by associating detection boxes whose scores are higher than a threshold. The objects with low detection scores, e.g. occluded objects, are simply thrown away, which brings non-negligible true object missing and fragmented trajectories. To solve this problem, we present a simple, effective and generic association method, tracking by associating every detection box instead of only the high score ones. For the low score detection boxes, we utilize their similarities with tracklets to recover true objects and filter out the background detections. When applied to 9 different state-of-the-art trackers, our method achieves consistent improvement on IDF1 score ranging from 1 to 10 points. To put forwards the state-of-the-art performance of MOT, we design a simple and strong tracker, named ByteTrack. For the first time, we achieve 80.3 MOTA, 77.3 IDF1 and 63.1 HOTA on the test set of MOT17 with 30 FPS running speed on a single V100 GPU.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/26813582/147467498-b8d16d8c-8472-4830-8bac-b107c49f7c6f.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```

## Results and models on MOT17

Please note that the performance on `MOT17-half-val` is comparable with the performance reported in the manuscript, while the performance on `MOT17-test` is lower than the performance reported in the manuscript.

The reason is that ByteTrack tunes customized hyper-parameters (e.g., image resolution and the high threshold of detection score) for each video in `MOT17-test` set, while we use unified parameters.

|  Method   | Detector |           Train Set           |    Test Set    | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                         Config                                          |                                                                                                                                                           Download                                                                                                                                                           |
| :-------: | :------: | :---------------------------: | :------------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :-------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ByteTrack | YOLOX-X  | CrowdHuman + MOT17-half-train | MOT17-half-val |   N    |       -        | 67.5 | 78.6 | 78.5 | 12852 | 21060 |  672  | [config](bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py) | [model](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500.log.json) |
| ByteTrack | YOLOX-X  | CrowdHuman + MOT17-half-train |   MOT17-test   |   N    |       -        | 61.7 | 78.1 | 74.8 | 36705 | 85032 | 2049  |  [config](bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17test.py)   | [model](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500.log.json) |

## Results and models on MOT20

Since there are only 4 videos in `MOT20-train`, ByteTrack is validated on `MOT17-train` rather than `MOT20-half-train`.

Please note that the MOTA on `MOT20-test` is slightly lower than that reported in the manuscript, because we don't tune the threshold for each video.

|  Method   | Detector |        Train Set         |  Test Set   | Public | Inf time (fps) | HOTA | MOTA | IDF1 |   FP   |   FN   | IDSw. |                                      Config                                      |                                                                                                                                                      Download                                                                                                                                                      |
| :-------: | :------: | :----------------------: | :---------: | :----: | :------------: | :--: | :--: | :--: | :----: | :----: | :---: | :------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ByteTrack | YOLOX-X  | CrowdHuman + MOT20-train | MOT17-train |   N    |       -        | 57.3 | 64.9 | 71.8 | 33,747 | 83,385 | 1,263 | [config](bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot20train_test-mot20test.py) | [model](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040-9ce38a60.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040.log.json) |
| ByteTrack | YOLOX-X  | CrowdHuman + MOT20-train | MOT20-test  |   N    |       -        | 61.5 | 77.0 | 75.4 | 33,083 | 84,433 | 1,345 | [config](bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot20train_test-mot20test.py) | [model](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040-9ce38a60.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot20-private_20220506_101040.log.json) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

**3.1 Joint training and tracking**

Some algorithm like ByteTrack, OCSORT don't need reid model, so we provide joint training and tracking for convenient.

```shell
# Training Bytetrack on crowdhuman and mot17-half-train dataset with following command
# The number after config file represents the number of GPUs used. Here we use 8 GPUs
bash tools/dist_train.sh configs/bytetrack/bytetrack_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8
```

**3.2 Separate training and tracking**

Of course, we provide train detector independently like SORT, DeepSORT, StrongSORT. Then use this detector to track.

```shell
# Training Bytetrack on crowdhuman and mot17-half-train dataset with following command
# The number after config file represents the number of GPUs used. Here we use 8 GPUs
bash tools/dist_train.sh configs/bytetrack/yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 4. Testing and evaluation

### 4.1 Example on MOTxx-halfval dataset

**4.1.1 use joint trained detector to evaluating and testing**

```shell
bash tools/dist_test_tracking.sh configs/bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8 --checkpoint ${CHECKPOINT_FILE}
```

**4.1.2 use separate trained detector to evaluating and testing**

```shell
bash tools/dist_test_tracking.sh configs/bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8 --detector ${CHECKPOINT_FILE}
```

**4.1.3 use video_baesd to evaluating and testing**

we also provide two_ways(img_based or video_based) to evaluating and testing.
if you want to use video_based to evaluating and testing, you can modify config as follows

```
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False))
```

#### 4.2 Example on MOTxx-test dataset

If you want to get the results of the [MOT Challenge](https://motchallenge.net/) test set, please use the following command to generate result files that can be used for submission. It will be stored in `./mot_17_test_res`, you can modify the saved path in `test_evaluator` of the config.

```shell
bash tools/dist_test_tracking.sh configs/bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17test.py 8 --checkpoint ${CHECKPOINT_FILE}
```

If you want to know about more detailed usage of `test_tracking.py/dist_test_tracking.sh/slurm_test_tracking.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 5.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/bytetrack/bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval.py --checkpoint ${CHECKPOINT_FILE} --out mot.mp4
```

If you want to know about more detailed usage of `mot_demo.py`, please refer to this [document](../../docs/en/user_guides/tracking_inference.md).

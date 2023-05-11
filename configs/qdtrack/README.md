# Quasi-Dense Similarity Learning for Multiple Object Tracking

## Abstract

<!-- [ABSTRACT] -->

Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can directly combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacementregression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QD-Track outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/48645550/158332287-79fb379b-d817-4aa8-8530-5f9d172b3ca7.png"/>
  <img src="https://user-images.githubusercontent.com/48645550/158332524-8ccaab0e-d379-4c6b-83e5-d75398af02bf.png"/>
</div>

## Results and models on MOT17

| Method  |   Detector   | Train Set  | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN   | IDSw. |                                      Config                                       |                                                                                                                                        Download                                                                                                                                        |
| :-----: | :----------: | :--------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :---: | :---: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QDTrack | Faster R-CNN | half-train | half-val |   N    |       -        | 57.1 | 68.1 | 68.6 | 7707 | 42732 | 1083  | [config](qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py) | [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635.log.json) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

```shell
# Training QDTrack on mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_train.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 4. Testing and evaluation

**4.1 Example on MOTxx-halfval dataset**

```shell
# Example 1: Test on motXX-half-val set
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_test_tracking.sh configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py 8 --checkpoint ${CHECKPOINT_PATH}
```

**4.2 use video_baesd to evaluating and testing**
we also provide two_ways(img_based or video_based) to evaluating and testing.
if you want to use video_based to evaluating and testing, you can modify config as follows

```
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False))
```

If you want to know about more detailed usage of `test_tracking.py/dist_test_tracking.sh/slurm_test_tracking.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 5.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py --checkpoint ${CHECKPOINT_PATH} --out mot.mp4
```

If you want to know about more detailed usage of `mot_demo.py`, please refer to this [document](../../docs/en/user_guides/tracking_inference.md).

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{pang2021quasi,
  title={Quasi-dense similarity learning for multiple object tracking},
  author={Pang, Jiangmiao and Qiu, Linlu and Li, Xia and Chen, Haofeng and Li, Qi and Darrell, Trevor and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={164--173},
  year={2021}
}
```

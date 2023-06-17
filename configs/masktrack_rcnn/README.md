# Video Instance Segmentation

## Abstract

<!-- [ABSTRACT] -->

In this paper we present a new computer vision task, named video instance segmentation. The goal of this new task is simultaneous detection, segmentation and tracking of instances in videos. In words, it is the first time that the image instance segmentation problem is extended to the video domain. To facilitate research on this new task, we propose a large-scale benchmark called YouTube-VIS, which consists of 2883 high-resolution YouTube videos, a 40-category label set and 131k high-quality instance masks. In addition, we propose a novel algorithm called MaskTrack R-CNN for this task. Our new method introduces a new tracking branch to Mask R-CNN to jointly perform the detection, segmentation and tracking tasks simultaneously. Finally, we evaluate the proposed method and several strong baselines on our new dataset. Experimental results clearly demonstrate the advantages of the proposed algorithm and reveal insight for future improvement. We believe the video instance segmentation task will motivate the community along the line of research for video understanding.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142986554-4f6a2630-92bc-43b4-8509-5173be00402d.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{yang2019video,
  title={Video instance segmentation},
  author={Yang, Linjie and Fan, Yuchen and Xu, Ning},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5188--5197},
  year={2019}
}
```

## Results and models of MaskTrack R-CNN on YouTube-VIS 2019 validation dataset

As mentioned in [Issues #6](https://github.com/youtubevos/MaskTrackRCNN/issues/6#issuecomment-502503505) in MaskTrack R-CNN, the result is kind of unstable for different trials, which ranges from 28 AP to 31 AP when using R-50-FPN as backbone.
The checkpoint provided below is the best one from two experiments.

|     Method      | Base detector | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  |                                 Config                                 |                                                                                                                                                                                    Download                                                                                                                                                                                    |
| :-------------: | :-----------: | :-------: | :-----: | :-----: | :------: | :------------: | :--: | :--------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MaskTrack R-CNN |  Mask R-CNN   | R-50-FPN  | pytorch |   12e   |   1.61   |       -        | 30.2 | [config](masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py)  |   [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830.log.json)   |
| MaskTrack R-CNN |  Mask R-CNN   | R-101-FPN | pytorch |   12e   |   2.27   |       -        | 32.2 | [config](masktrack-rcnn_mask-rcnn_r101_fpn_8xb1-12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038-454dc48b.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038.log.json) |
| MaskTrack R-CNN |  Mask R-CNN   | X-101-FPN | pytorch |   12e   |   3.69   |       -        | 34.7 | [config](masktrack-rcnn_mask-rcnn_x101_fpn_8xb1-12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205-fff7a102.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205.log.json) |

## Results and models of MaskTrack R-CNN on YouTube-VIS 2021 validation dataset

The checkpoint provided below is the best one from two experiments.

|     Method      | Base detector | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  |                                 Config                                 |                                                                                                                                                                                    Download                                                                                                                                                                                    |
| :-------------: | :-----------: | :-------: | :-----: | :-----: | :------: | :------------: | :--: | :--------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MaskTrack R-CNN |  Mask R-CNN   | R-50-FPN  | pytorch |   12e   |   1.61   |       -        | 28.7 | [config](masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2021.py)  |   [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948-10da90d9.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948.log.json)   |
| MaskTrack R-CNN |  Mask R-CNN   | R-101-FPN | pytorch |   12e   |   2.27   |       -        | 31.3 | [config](masktrack-rcnn_mask-rcnn_r101_fpn_8xb1-12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509-3c49e4f3.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509.log.json) |
| MaskTrack R-CNN |  Mask R-CNN   | X-101-FPN | pytorch |   12e   |   3.69   |       -        | 33.5 | [config](masktrack-rcnn_mask-rcnn_x101_fpn_8xb1-12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943-90831df4.pth) \| [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943.log.json) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

```shell
# Training MaskTrack R-CNN on YouTube-VIS-2021 dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_train.sh configs/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2021.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 4. Testing and evaluation

If you want to get the results of the [YouTube-VOS](https://youtube-vos.org/dataset/vis/) val/test set, please use the following command to generate result files that can be used for submission. It will be stored in `./youtube_vis_results.submission_file.zip`, you can modify the saved path in `test_evaluator` of the config.

```shell
# The number after config file represents the number of GPUs used.
bash tools/dist_test_tracking.sh configs/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2021.py 8 --checkpoint ${CHECKPOINT_PATH}
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 5.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2021.py  --checkpoint {CHECKPOINT_PATH} --out vis.mp4
```

If you want to know about more detailed usage of `mot_demo.py`, please refer to this [document](../../docs/en/user_guides/tracking_inference.md).

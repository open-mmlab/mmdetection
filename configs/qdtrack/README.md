# Quasi-Dense Similarity Learning for Multiple Object Tracking

## Abstract

<!-- [ABSTRACT] -->

Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can directly combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacementregression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QD-Track outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/48645550/158332287-79fb379b-d817-4aa8-8530-5f9d172b3ca7.png"/>
  <img src="https://user-images.githubusercontent.com/48645550/158332524-8ccaab0e-d379-4c6b-83e5-d75398af02bf.png"/>
</div>

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

## Results and models on MOT17

| Method  |   Detector   |        Train Set        | Test Set | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP  |  FN   | IDSw. |                                            Config                                            |                                                                                                                                                   Download                                                                                                                                                   |
| :-----: | :----------: | :---------------------: | :------: | :----: | :------------: | :--: | :--: | :--: | :--: | :---: | :---: | :------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QDTrack | Faster R-CNN |       half-train        | half-val |   N    |       -        | 56.3 | 66.7 | 67.9 | 9054 | 43668 | 1125  |      [config](qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py)       |            [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635.log.json)            |
| QDTrack | Faster R-CNN | CrowdHuman + half-train | half-val |   N    |       -        | 58.1 | 69.8 | 70.4 | 7281 | 40500 | 1050  | [config](qdtrack_faster-rcnn_r50_fpn_8xb2-4e_crowdhuman-mot17halftrain_test-mot17halfval.py) | [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453.log.json) |

## Results and models on LVIS dataset

| Method  |   Detector   |     Train Set     |    Test Set    | Inf time (fps) |  AP  | AP50 | AP75 | AP_S | AP_M | AP_L |                              Config                              |                                                                                                                                         Download                                                                                                                                         |
| :-----: | :----------: | :---------------: | :------------: | :------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QDTrack | Faster R-CNN | LVISv0.5+COCO2017 | TAO validation |       -        | 17.2 | 28.6 | 17.7 | 5.3  | 13.0 | 22.1 | [config](qdtrack_faster-rcnn_r101_fpn_8xb2-24e_lvis_test-tao.py) | [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/tao_dataset/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513-88911daf.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/tao_dataset/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513.log.json) |

## Results and models on TAO dataset

Note: If you want to achieve a track AP of 11.0 on the TAO dataset, you need to do pre-training on LVIS dataset.

a. Pre-train the QDTrack on LVISv0.5+COCO2017 training set and save the model to `checkpoints/lvis/**.pth`.

The pre-trained checkpoint is given above([model](https://download.openmmlab.com/mmtracking/mot/qdtrack/tao_dataset/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513-88911daf.pth)).

b. Modify the configs for TAO accordingly(set `load_from` to your **ckpt path**).

See `1.2 Example on TAO Dataset` to get more details.

We observe around 0.5 track AP fluctuations in performance, and provide the best model.

| Method  |   Detector   | Train Set |    Test Set    | Inf time (fps) | Track AP(50:75) | Track AP50 | Track AP75 |                         Config                         |                                                                                                                                        Download                                                                                                                                        |
| :-----: | :----------: | :-------: | :------------: | :------------: | :-------------: | :--------: | :--------: | :----------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QDTrack | Faster R-CNN | TAO train | TAO validation |       -        |      11.0       |    15.8    |    6.1     | [config](qdtrack_faster-rcnn_r101_fpn_8xb2-12e_tao.py) | [model](https://download.openmmlab.com/mmtracking/mot/qdtrack/tao_dataset/qdtrack_faster-rcnn_r101_fpn_12e_tao_20220613_211934-7cbf4062.pth) \| [log](https://download.openmmlab.com/mmtracking/mot/qdtrack/tao_dataset/qdtrack_faster-rcnn_r101_fpn_12e_tao_20220613_211934.log.json) |

## Get started

### 1. Training

Due to the influence of parameters such as learning rate in default configuration file, we recommend using 8 GPUs for training in order to reproduce accuracy. You can use the following command to start the training.

**1.1 Example on MOT Challenge Dataset**

```shell
# Training QDTrack on crowdhuman and mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_train.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn-r50_fpn_8xb2-4e_crowdhuman-mot17halftrain_test-mot17halfval.py 8
```

**1.2 Example on TAO Dataset**

- a. Pre-train the QDTrack on LVISv0.5+COCO2017 training set and save the model to `checkpoints/lvis/**.pth`.

```shell
./tools/dist_train.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_8xb2-24e_lvis_test-tao.py 8
```

- b. Modify the configs for TAO accordingly(set `load_from` to your **ckpt path**).

```shell
./tools/dist_train.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_8xb2-12e_tao.py 8 \
    --cfg-options load_from=checkpoints/lvis/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513-88911daf.pth
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 2. Testing and evaluation

**2.1 Example on MOTxx-halfval dataset**

```shell
# Example 1: Test on motXX-half-val set
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
./tools/dist_test.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn-r50_fpn_8xb2-4e_crowdhuman-mot17halftrain_test-mot17halfval.py 8 \
    --checkpoint ./checkpoints/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth
```

**2.2 Example on TAO dataset**

Note that the previous section `Results and models on TAO dataset` is evaluated using this command.

```shell
# Example 2: Test on TAO dataset
# The number after config file represents the number of GPUs used.
./tools/dist_test.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_8xb2-12e_tao.py 8 \
    --checkpoint ./checkpoints/qdtrack_faster-rcnn_r101_fpn_12e_tao_20220613_211934-7cbf4062.pth
```

In addition, you can use the following command to check the results of the previous section `Results and models on LVIS dataset`.

```shell
# Please note that their test sets are the same, only the training sets are different.
./tools/dist_test.sh \
    configs/mot/qdtrack/qdtrack_faster-rcnn_r101_fpn_8xb2-24e_lvis_test-tao.py 8 \
    --checkpoint ./checkpoints/qdtrack_faster-rcnn_r101_fpn_24e_lvis_20220430_024513-88911daf.pth
```

If you want to know about more detailed usage of `test.py/dist_test.sh/slurm_test.sh`, please refer to this [document](../../../docs/en/user_guides/4_train_test.md).

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/demo_mot_vis.py \
    configs/mot/qdtrack/qdtrack_faster-rcnn-r50_fpn_8xb2-4e_crowdhuman-mot17halftrain_test-mot17halfval.py \
    --checkpoint ./checkpoints/qdtrack_faster-rcnn_r50_fpn_4e_crowdhuman_mot17_20220315_163453-68899b0a.pth \
    --input demo/demo.mp4 \
    --output mot.mp4
```

If you want to know about more detailed usage of `demo_mot_vis.py`, please refer to this [document](../../../docs/en/user_guides/3_inference.md).

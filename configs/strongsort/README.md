# StrongSORT: Make DeepSORT Great Again

## Abstract

<!-- [ABSTRACT] -->

Existing Multi-Object Tracking (MOT) methods can be roughly classified as tracking-by-detection and joint-detection-association paradigms. Although the latter has elicited more attention and demonstrates comparable performance relative to the former, we claim that the tracking-by-detection paradigm is still the optimal solution in terms of tracking accuracy. In this paper, we revisit the classic tracker DeepSORT and upgrade it from various aspects, i.e., detection, embedding and association. The resulting tracker, called StrongSORT, sets new HOTA and IDF1 records on MOT17 and MOT20. We also present two lightweight and plug-and-play algorithms to further refine the tracking results. Firstly, an appearance-free link model (AFLink) is proposed to associate short tracklets into complete trajectories. To the best of our knowledge, this is the first global link model without appearance information. Secondly, we propose Gaussian-smoothed interpolation (GSI) to compensate for missing detections. Instead of ignoring motion information like linear interpolation, GSI is based on the Gaussian process regression algorithm and can achieve more accurate localizations. Moreover, AFLink and GSI can be plugged into various trackers with a negligible extra computational cost (591.9 and 140.9 Hz, respectively, on MOT17). By integrating StrongSORT with the two algorithms, the final tracker StrongSORT++ ranks first on MOT17 and MOT20 in terms of HOTA and IDF1 metrics and surpasses the second-place one by 1.3 - 2.2. Code will be released soon.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/99722489/185282811-ec82bdf6-8889-4f01-9c4d-a8e104f775b7.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```

## Results and models on MOT17

|    Method    | Detector | ReID |           Train Set           |    Test Set    | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                        Config                                        |                                                                                                                                                                                   Download                                                                                                                                                                                    |
| :----------: | :------: | :--: | :---------------------------: | :------------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :----------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| StrongSORT++ | YOLOX-X  | R50  | CrowdHuman + MOT17-half-train | MOT17-half-val |   N    |       -        | 70.9 | 78.4 | 83.3 | 15237 | 19035 |  582  | [config](strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py) | [detector](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot17-private-half_20220812_192036-b6c9ce9a.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot17-4bf6b63d.pth) [AFLink](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth) |

## Results and models on MOT20

|    Method    | Detector | ReID |        Train Set         |  Test Set  | Public | Inf time (fps) | HOTA | MOTA | IDF1 |  FP   |  FN   | IDSw. |                                    Config                                     |                                                                                                                                                                                         Download                                                                                                                                                                                         |
| :----------: | :------: | :--: | :----------------------: | :--------: | :----: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :---------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| StrongSORT++ | YOLOX-X  | R50  | CrowdHuman + MOT20-train | MOT20-test |   N    |       -        | 62.9 | 75.5 | 77.3 | 29043 | 96155 | 1640  | [config](strongsort_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test.py) | [detector](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/yolox_x_crowdhuman_mot20-private_20220812_192123-77c014de.pth) [reid](https://download.openmmlab.com/mmtracking/mot/reid/reid_r50_6e_mot20_20210803_212426-c83b1c01.pth) [AFLink](https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth) |

## Get started

### 1. Development Environment Setup

Tracking Development Environment Setup can refer to this [document](../../docs/en/get_started.md).

### 2. Dataset Prepare

Tracking Dataset Prepare can refer to this [document](../../docs/en/user_guides/tracking_dataset_prepare.md).

### 3. Training

We implement StrongSORT with independent detector and ReID models.
Note that, due to the influence of parameters such as learning rate in default configuration file,
we recommend using 8 GPUs for training in order to reproduce accuracy.

You can train the detector as follows.

```shell script
# Training YOLOX-X on crowdhuman and mot17-half-train dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_train.sh configs/det/yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8
```

And you can train the ReID model as follows.

```shell script
# Training ReID model on mot17-train80 dataset with following command.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_train.sh configs/reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py 8
```

If you want to know about more detailed usage of `train.py/dist_train.sh/slurm_train.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 4. Testing and evaluation

**2.1 Example on MOTxx-halfval dataset**

```shell script
# Example 1: Test on motXX-half-val set.
# The number after config file represents the number of GPUs used. Here we use 8 GPUs.
bash tools/dist_test_tracking.sh configs/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8 --detector ${CHECKPOINT_PATH} --reid ${CHECKPOINT_PATH}
```

**2.2 Example on MOTxx-test dataset**

If you want to get the results of the [MOT Challenge](https://motchallenge.net/) test set,
please use the following command to generate result files that can be used for submission.
It will be stored in `./mot_20_test_res`, you can modify the saved path in `test_evaluator` of the config.

```shell script
# Example 2: Test on motxx-test set
# The number after config file represents the number of GPUs used
bash tools/dist_test_tracking.sh configs/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot20train_test-mot20test.py 8 --detector ${CHECKPOINT_PATH} --reid ${CHECKPOINT_PATH}
```

If you want to know about more detailed usage of `test_tracking.py/dist_test_tracking.sh/slurm_test_tracking.sh`,
please refer to this [document](../../docs/en/user_guides/tracking_train_test.md).

### 3.Inference

Use a single GPU to predict a video and save it as a video.

```shell
python demo/mot_demo.py demo/demo_mot.mp4 configs/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py --detector ${CHECKPOINT_FILE} --reid ${CHECKPOINT_PATH} --out mot.mp4
```

If you want to know about more detailed usage of `mot_demo.py`, please refer to this [document](../../docs/en/user_guides/tracking_inference.md).

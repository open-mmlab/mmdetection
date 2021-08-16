## Introduction

This project is trying to Recreate CenternetV2 based on MMDET, which is proposed in paper [Probabilistic two-stage detection](https://arxiv.org/pdf/2103.07461.pdf). 

This project is also for the contest [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021).

This is NOT the official implementation.

Quick peek at the result:

> **The paper:** centernet2(CascadeRCNN-CenterNet w. prob.) mAP is 42.9.
>
> **This implementation:** we implement centernet2(CascadeRCNN-CenterNet w. prob.) mAP is 41.9.
> 
> **Reason:** There are problems with learning strategies.
> 
> **Note:** We will continue to maintain the code to get the map to 42.9.
> 
> **Note:** We then we're going to reproduce the weights, which Verification and inference part.

## Implementation

**Code:** we add detector in mmdet/models/detectors/centernetv2.py.

**Code:** we add config in configs/centernetv2/centernet2.py.

**Code:** we add centernet_head in mmdet/dense_heads/centernet_headv2.py.

**Code:** we add hm_binary_focal_loss in mmdet/losses/hm_binary_focal_loss.py.

**Code:** we modify mmdet/roi_heads/cascade_roi_head.py, add the super parameter add_agnostic_score to Control whether the first stage score.This hyperparameter does not affect the use of other configuration files

**Code:** we modify mmdet/bbox_head/bbox_head.py, add the super parameter add_agnostic_score to Control Whether to use softmax.This hyperparameter does not affect the use of other configuration files

## Experiments

**MMDetection:** this project is based on version v2.13.0.

**MMCV:** version v1.3.5

**Dataset:** coco_train_2017(117k) as training dataset and coco_val_2017(5k) as testing dataset. All the results are reported on coco_val_2017.

Results reported in the paper:

|                              | AP      | AP50   | AP75   | APs    | APm    | APl    |
| ---------------------------- | ------- | ------ | ------ | ------ | ------ | ------ |
| cascadeRCNN-CenterNet w.prob | 42.862  | 59.519 | 47.028 | 24.064 | 47.043 | 56.197 |


Results by this implementation:(8 k80)

|                              | AP     | AP50  | AP75   | APs    | APm    | APl    |
| ---------------------------- | ------ | ----- | ------ | ------ | ------ | ------ |
| cascadeRCNN-CenterNet w.prob | 41.9   | 59.0  | 46.4   | 24.4   | 45.6   | 55.4   |


Log and model:

|                      | backbone | Iter | bbox AP | Config                                                       | Log                                                          | Model                                                        | GPUs |
| -------------------- | -------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| cascadeRCNN-CenterNet w.prob | R-50-FPN | 9000      | 41.9    | [config](https://github.com/huimlight/mmdetection/blob/centernetv2/configs/centernetv2/centernet2.py) | [log](https://github.com/huimlight/mmdet-centernetv2/blob/main/log) | [baidu ](https://pan.baidu.com/s/1XACylKLLZhBKo8nUJubA7g) [1171] | 8    |


## Extra:

We fixed a clear bug from original CenterNet2 code, which produces repeated positive indices during heatmap loss calculation. And we also provided an optional distance map generation method (which differs from CenterNet2 in positive sample ranges). 

Here we report both the original version and revised version results, please note that both versions fixed the ‘duplicated positive indices’ bug and both are training on a single V100 GPU for 90k iterations using a batch size of 20.

Original Version:

Results by this implementation:(single v100)

|                              | AP     | AP50  | AP75   | APs    | APm    | APl    |
| ---------------------------- | ------ | ----- | ------ | ------ | ------ | ------ |
| cascadeRCNN-CenterNet w.prob | 42.8   | 60.5  | 47.7   | 25.4   | 45.9   | 56.0   |

Revised Version:

Results by this implementation:(single v100)

|                              | AP     | AP50  | AP75   | APs    | APm    | APl    |
| ---------------------------- | ------ | ----- | ------ | ------ | ------ | ------ |
| cascadeRCNN-CenterNet w.prob | 42.7   | 60.4  | 47.5   | 25.1   | 46.1   | 55.6   |

We found that during training process, the improved version showed a faster convergence and outplayed the original version in performance for the most time. However, the original version laughed at last and won the competition at the very last validation stage with a close match score 0.428 vs 0.427. We assume that the seemingly ‘unreasonable’ assign of positive indices may add the robustness of the model.

Log and model:

|                      | backbone | Iter | bbox AP | Config                                                       | Log                                                          | Model                                                        | GPUs |
| -------------------- | -------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| cascadeRCNN-CenterNet w.prob | R-50-FPN | 9000      | 42.8    | [config](https://github.com/huimlight/mmdet-centernetv2/blob/main/centernet2_res50_rpn_90k_coco.py) | [log](https://github.com/huimlight/mmdet-centernetv2/blob/main/20210810_192908.log.json) | [baidu ](https://pan.baidu.com/s/1lVwFiMqCe3BsgeRSdR9u_g) [5bot] | single v100    |

Log and model:

|                      | backbone | Iter | bbox AP | Config                                                       | Log                                                          | Model                                                        | GPUs |
| -------------------- | -------- | ------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| cascadeRCNN-CenterNet w.prob | R-50-FPN | 9000      | 42.7    | [config](https://github.com/huimlight/mmdet-centernetv2/blob/main/centernet2_res50_rpn_90k_coco_0.47.py) | [log](https://github.com/huimlight/mmdet-centernetv2/blob/main/20210810_192413.log.json) | [baidu ](https://pan.baidu.com/s/1kDOMnEpzFNjaxld8rHTCLw) [osix] | single v100    |


## Usage

You can train and inference the model like any other models in MMDetection, see [docs](https://mmdetection.readthedocs.io/) for details.


## Acknowledgement

[Probabilistic two-stage detection](https://arxiv.org/pdf/2103.07461.pdf)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMCV](https://github.com/open-mmlab/mmcv)

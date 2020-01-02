# Is Sampling Heuristics Necessary in Training Deep Object Detectors


## Introduction

```
@article{sampling_free,
author    = {Joya Chen and
             Dong Liu and
             Tong Xu and
             Shilong Zhang and
             Shiwei Wu and
             Bin Luo and
             Xuezheng Peng and
             Enhong Chen},
title     = {Is Sampling Heuristics Necessary in Training Deep Object Detectors?},
journal   = {CoRR},
volume    = {abs/1909.04868},
year      = {2019},
url       = {http://arxiv.org/abs/1909.04868},
archivePrefix = {arXiv},
eprint    = {1909.04868},
}
```


## Results and Models


### PASCAL VOC dataset

| Model                                                                                    | Style   | Lr schd | box AP          |
|:----------------------------------------------------------------------------------------:|:-------:|:-------:|:---------------:|
| Faster R-50                                                                              | pytorch | 1x      |  80.4           |
| `Faster R-50 + Sampling-Free`                                                            | pytorch | 1x      |  `81.4 (+1.0)`  |
| Faster R-50 [Official Repo](https://github.com/ChenJoya/sampling-free)                   | pytorch | 1x      |  80.9           |
| `Faster R-50 + Sampling-Free` [Official Repo](https://github.com/ChenJoya/sampling-free) | pytorch | 1x      |  `81.5 (+0.6)`  |



### COCO dataset

| Model                                                                                    | Style   | Lr schd | box AP          |
|:----------------------------------------------------------------------------------------:|:-------:|:-------:|:---------------:|
| Faster R-50                                                                              | pytorch | 1x      |  36.4           |
| `Faster R-50 + Sampling-Free`                                                            | pytorch | 1x      |  `37.8 (+1.4)`  |
| Faster R-50 [Official Repo](https://github.com/ChenJoya/sampling-free)                   | pytorch | 1x      |  36.8           |
| `Faster R-50 + Sampling-Free` [Official Repo](https://github.com/ChenJoya/sampling-free) | pytorch | 1x      |  `38.4 (+1.6)`  |


**Notes:**
- sample-free include `Bias Initialization, Guided Loss, Class-Adaptive Threshold`, to keep the origin mmdetection code structure, 
we compute prior and set score_thr before training, so for a specific detector and dataset, please run `configs/sample_free/sample_free.py` before training
- due to the hardware limit, we train the model use 2 GPUs, for pascal voc, we set `lr = 0.01 / 4` and `0.02 /4` for coco.
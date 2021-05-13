# CenterNet

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{zhou2019objects,
  title={Objects as Points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```

## Results and models

| Backbone        | DCN |  Mem (GB) | Inf time (fps) | Box AP | Flip box AP| Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :----: | :----: |
| ResNet-18 | N |  |  |  |  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_140e_coco.py) | [model]() &#124; [log]() |
| ResNet-18 | Y | 3.47 |  | 29.7 | 31.0 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centernet/centernet_resnet18_dcnv2_140e_coco.py) | [model]() &#124; [log]() |

Note:

- Flip box AP setting is single-scale and `flip=True`.
- Compared to the source code, we refer to [CenterNet-Better](https://github.com/FateScript/CenterNet-better), and make the following changes
  - fix mean and variance errors
  - use a different optimizer setting and warmup
  - use DDP while the original repo uses DP

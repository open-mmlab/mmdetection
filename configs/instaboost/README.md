# InstaBoost for MMDetection

Configs in this directory is implemented for ICCV2019 paper "InstaBoost: Boosting Instance Segmentation Via Probability Map Guided Copy-Pasting". InstaBoost is a data augmentation method for object detection and instance segmentation. Our paper has been released on [`arXiv`](https://arxiv.org/abs/1908.07801).

## Install InstaBoost
```
pip install instaboostfast
# in python
>>> import instaboostfast as instaboost
```

Our code and more details can be found [`here`](https://github.com/GothicAi/Instaboost).

## Implement InstaBoost In MMdetection

We have already integrate InstaBoost in data pipeline, thus all you need is to add or change **InstaBoost** configurations after **LoadImageFromFile** like [this line](mask_rcnn_r50_fpn_instaboost_4x.py#L121). You can refer to [`InstaBoostConfig`](https://github.com/GothicAi/InstaBoost-pypi#instaboostconfig) for details.

## Model Zoo

Results and models are available in the [Model zoo](MODEL_ZOO.md). More models are coming!

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{Fang2019InstaBoost,
author = {Fang, Hao-Shu and Sun, Jianhua and Wang, Runzhong and Gou, Minghao and Li, Yong-Lu and Lu, Cewu},
title = {InstaBoost: Boosting Instance Segmentation Via Probability Map Guided Copy-Pasting},
journal={arXiv preprint arXiv:1908.07801},
year = {2019}
}
```
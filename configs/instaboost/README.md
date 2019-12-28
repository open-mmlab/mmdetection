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

## Results and Models

 - All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`, for your conveinience of evaluation and comparison. In our paper, the numbers are obtained from test-dev.
 - To balance accuracy and training time when using InstaBoost, models released in this page are all trained for 48 Epochs. Other training and testing configs are strictly following the original framework. 
 - More results for other detection frameworks are avaliable [`here`](https://github.com/GothicAi/Instaboost).
 - More models are coming!

|     Network     |       Backbone       | Lr schd |      box AP       |      mask AP       |      Download       |
| :-------------: |      :--------:      | :-----: |      :----:       |      :-----:       | :-----------------: |
|    Mask R-CNN   |       R-50-FPN       |   4x    |  39.90(orig:37.3)  |  36.20(orig:34.2)   |[Baidu]() / [Google]()|

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
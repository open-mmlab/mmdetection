# InstaBoost for MMDetection

<!-- [ALGORITHM] -->

Configs in this directory is the implementation for ICCV2019 paper "InstaBoost: Boosting Instance Segmentation Via Probability Map Guided Copy-Pasting" and provided by the authors of the paper. InstaBoost is a data augmentation method for object detection and instance segmentation. The paper has been released on [`arXiv`](https://arxiv.org/abs/1908.07801).

```latex
@inproceedings{fang2019instaboost,
  title={Instaboost: Boosting instance segmentation via probability map guided copy-pasting},
  author={Fang, Hao-Shu and Sun, Jianhua and Wang, Runzhong and Gou, Minghao and Li, Yong-Lu and Lu, Cewu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={682--691},
  year={2019}
}
```

## Usage

### Requirements

You need to install `instaboostfast` before using it.

```shell
pip install instaboostfast
```

The code and more details can be found [here](https://github.com/GothicAi/Instaboost).

### Integration with MMDetection

InstaBoost have been already integrated in the data pipeline, thus all you need is to add or change **InstaBoost** configurations after **LoadImageFromFile**. We have provided examples like [this](mask_rcnn_r50_fpn_instaboost_4x#L121). You can refer to [`InstaBoostConfig`](https://github.com/GothicAi/InstaBoost-pypi#instaboostconfig) for more details.

## Results and Models

- All models were trained on `coco_2017_train` and tested on `coco_2017_val` for convenience of evaluation and comparison. In the paper, the results are obtained from `test-dev`.
- To balance accuracy and training time when using InstaBoost, models released in this page are all trained for 48 Epochs. Other training and testing configs strictly follow the original framework.
- For results and models in MMDetection V1.x, please refer to [Instaboost](https://github.com/GothicAi/Instaboost).

|     Network     |       Backbone       | Lr schd | Mem (GB) | Inf time (fps) | box AP  | mask AP | Config |     Download       |
| :-------------: |      :--------:      | :-----: | :------: | :------------: | :------:| :-----: | :------: | :-----------------: |
|    Mask R-CNN   |       R-50-FPN       |   4x    | 4.4      | 17.5           | 40.6    | 36.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307_223635.log.json) |
|    Mask R-CNN   |      R-101-FPN       |   4x    | 6.4       |                | 42.5    | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738.log.json) |
|    Mask R-CNN   |   X-101-64x4d-FPN    |   4x    | 10.7     |                | 44.7    | 39.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947.log.json) |
|  Cascade R-CNN  |       R-101-FPN      |   4x    | 6.0      | 12.0            | 43.7    | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-c19d98d9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307_223646.log.json) |

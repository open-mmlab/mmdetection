# Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the instance segmentation performance in the CVPR 2021 paper for [Seesaw Loss for Long-Tailed Instance Segmentation](https://arxiv.org/abs/2008.10032).

```latex
@inproceedings{gupta2019lvis,
  title={Seesaw Loss for Long-Tailed Instance Segmentation},
  author={Jiaqi Wang and Wenwei Zhang and Yuhang Zang and Yuhang Cao and Jiangmiao Pang and Tao Gong and Kai Chen and Ziwei Liu and Chen Change Loy and Dahua Lin},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```


* Please setup [LVIS dataset](../lvis/README.md) for MMDetection.

* RFS indicates to use oversample strategy [here](../../docs/tutorials/new_dataset.md#class-balanced-dataset) with oversample threshold `1e-3`.

## Results and models of Seasaw Loss on LVIS v1 dataset


|       Method       | Backbone  |  Style  | Lr schd | Data Sampler | Norm Mask | box AP | mask AP |                                                                                 Config                                                                                  |         Download         |
| :----------------: | :-------: | :-----: | :-----: | :----------: | :-------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: |
|     Mask R-CNN     | R-50-FPN  | pytorch |   2x    |    random    |     N     |  25.6  |  25.0   |             [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py)              | [model]() &#124; [log]() |
|     Mask R-CNN     | R-50-FPN  | pytorch |   2x    |    random    |     Y     |  25.6  |  25.4   |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py)        | [model]() &#124; [log]() |
|     Mask R-CNN     | R-101-FPN | pytorch |   2x    |    random    |     N     |  27.4  |  26.7   |                   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py)                   | [model]() &#124; [log]() |
|     Mask R-CNN     | R-101-FPN | pytorch |   2x    |    random    |     Y     |  27.2  |  27.3   |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py)       | [model]() &#124; [log]() |
|     Mask R-CNN     | R-50-FPN  | pytorch |   2x    |     RFS      |     N     |  27.6  |  26.4   |           [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py)            | [model]() &#124; [log]() |
|     Mask R-CNN     | R-50-FPN  | pytorch |   2x    |     RFS      |     Y     |  27.6  |  26.8   |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py)      | [model]() &#124; [log]() |
|     Mask R-CNN     | R-101-FPN | pytorch |   2x    |     RFS      |     N     |  28.9  |  27.6   |           [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py)           | [model]() &#124; [log]() |
|     Mask R-CNN     | R-101-FPN | pytorch |   2x    |     RFS      |     Y     |  28.9  |  28.2   |     [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py)     | [model]() &#124; [log]() |
| Cascade Mask R-CNN | R-101-FPN | pytorch |   2x    |    random    |     N     |  33.1  |  29.2   |         [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py)         | [model]() &#124; [log]() |
| Cascade Mask R-CNN | R-101-FPN | pytorch |   2x    |    random    |     Y     |  33.0  |  30.0   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py)   | [model]() &#124; [log]() |
| Cascade Mask R-CNN | R-101-FPN | pytorch |   2x    |     RFS      |     N     |  30.0  |  29.3   |       [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py)       | [model]() &#124; [log]() |
| Cascade Mask R-CNN | R-101-FPN | pytorch |   2x    |     RFS      |     Y     |  32.8  |  30.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py) | [model]() &#124; [log]() |

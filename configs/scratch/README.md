# Scratch

> [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883)

<!-- [ALGORITHM] -->

## Abstract

We report competitive results on object detection and instance segmentation on the COCO dataset using standard models trained from random initialization. The results are no worse than their ImageNet pre-training counterparts even when using the hyper-parameters of the baseline system (Mask R-CNN) that were optimized for fine-tuning pre-trained models, with the sole exception of increasing the number of training iterations so the randomly initialized models may converge. Training from random initialization is surprisingly robust; our results hold even when: (i) using only 10% of the training data, (ii) for deeper and wider models, and (iii) for multiple tasks and metrics. Experiments show that ImageNet pre-training speeds up convergence early in training, but does not necessarily provide regularization or improve final target task accuracy. To push the envelope we demonstrate 50.9 AP on COCO object detection without using any external data---a result on par with the top COCO 2017 competition results that used ImageNet pre-training. These observations challenge the conventional wisdom of ImageNet pre-training for dependent tasks and we expect these discoveries will encourage people to rethink the current de facto paradigm of \`pre-training and fine-tuning' in computer vision.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143974572-69c4f57d-0d6d-4f56-ba91-23f8a65a2a77.png" height="300"/>
</div>

## Results and Models

|    Model     | Backbone |  Style  | Lr schd | box AP | mask AP |                                                            Config                                                             |                                                                                                                                                                                 Download                                                                                                                                                                                  |
| :----------: | :------: | :-----: | :-----: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Faster R-CNN | R-50-FPN | pytorch |   6x    |  40.7  |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py) |     [model](https://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_20200201_193013.log.json)     |
|  Mask R-CNN  | R-50-FPN | pytorch |   6x    |  41.2  |  37.4   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco.py)  | [model](https://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_20200201_193051.log.json) |

Note:

- The above models are trained with 16 GPUs.

## Citation

```latex
@article{he2018rethinking,
  title={Rethinking imagenet pre-training},
  author={He, Kaiming and Girshick, Ross and Doll{\'a}r, Piotr},
  journal={arXiv preprint arXiv:1811.08883},
  year={2018}
}
```

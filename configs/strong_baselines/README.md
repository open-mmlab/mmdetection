# Strong Baselines

<!-- [OTHERS] -->

We train Mask R-CNN with large-scale jitter and longer schedule as strong baselines.
The modifications follow those in [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/configs/new_baselines).

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   50e   |          |                |        |         |  [config](./mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_50e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   100e  |          |                |        |         |  [config](./mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | caffe |   100e  |          |                |   44.7  |  40.4   |  [config](./mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | caffe |   400e  |          |                |        |         |  [config](./mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_400e_coco.py) | [model]() &#124; [log]() |

## Notice

When using large-scale jittering, there are sometimes empty proposals in the box and mask heads during training.
This requires MMSyncBN that allows empty tensors. Therefore, please use mmcv-full>=1.3.14 to train models supported in this directory.

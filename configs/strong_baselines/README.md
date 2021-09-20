# Strong Baselines

We train Mask R-CNN with large-scale jittor and longer schedule as strong baselines.
The modifications follow those in [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/configs/new_baselines).

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   50e   |          |                |        |         |  [config](./mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_50e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   100e  |          |                |        |         |  [config](./mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | caffe |   100e  |          |                |   44.9  |  40.5   |  [config](./mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco.py) | [model]() &#124; [log]() |
|    R-50-FPN     | caffe |   400e  |          |                |        |         |  [config](./mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_400e_coco.py) | [model]() &#124; [log]() |

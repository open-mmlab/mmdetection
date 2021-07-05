# Strong Baselines

We train Mask R-CNN with large-scale jittor and longer schedule as strong baselines.
The modifications follow those in [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/configs/new_baselines).


## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   50e   |          |                |        |         |  [config]() | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   100e  |          |                |        |         |  [config]() | [model]() &#124; [log]() |
|    R-50-FPN     | pytorch |   200e  |          |                |        |         |  [config]() | [model]() &#124; [log]() |

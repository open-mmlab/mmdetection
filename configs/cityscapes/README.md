## Common settings

- All baselines were trained using 8 GPU with a batch size of 8 (1 images per GPU) using the [linear scaling rule](https://arxiv.org/abs/1706.02677) to scale the learning rate. 
- All models were trained on `cityscapes_train`, and tested on `cityscapes_val`.
- 1x training schedule indicates 64 epochs which corresponds to slightly less than the 24k iterations reported in the original schedule from the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.


## Baselines

Download links and more models with different backbones and training schemes will be added to the model zoo.


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :---:    | :------: | :-----------------: | :------------: | :----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | 4.9      | 0.345               | 8.8            | 36.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/cityscapes/faster_rcnn_r50_fpn_1x_city_20190727-7b9c0534.pth) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :------: | :------: | :-----------------: | :------------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | 4.9      | 0.609               | 2.5            | 37.4  |  32.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/cityscapes/mask_rcnn_r50_fpn_1x_city_20190727-9b3c56a5.pth) |

**Notes:**
- In the original paper, the mask AP of Mask R-CNN R-50-FPN is 31.5.


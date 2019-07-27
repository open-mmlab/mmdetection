# Benchmark and Model Zoo

## Environment

### Hardware

- 1 NVIDIA GeForce GTX 1080 Ti GPU

## Common settings

- All baselines were trained using 1 GPU with a batch size of 2 (2 images per GPU) using the [linear scaling rule](https://arxiv.org/abs/1706.02677) to scale the learning rate. The learning rate in the configs is set for a batch size of 16 to match the default of the coco models.
- All models were trained on `cityscapes_train`, and tested on `cityscapes_val`.
- 1x training schedule indicates 64 epochs which corresponds to slightly less than the 24k iterations reported in the original schedule from the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.


## Baselines

Download links and more models with different backbones and training schemes will be added to the model zoo.


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Pretraining | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :---:    | :---------: | :----:   | :----:              | :----:          | :----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | Backbone    | 4.9      | 0.345               | 8.8            | 36.0   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/cityscapes/faster_rcnn_r50_fpn_1x_city_20190727-7b9c0534.pth) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Pretraining | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :---:    | :---------: | :----:   | :----:              | :----:           | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | Backbone    | 4.9      | 0.609               | 2.5            | 37.4  |  32.5   | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/cityscapes/mask_rcnn_r50_fpn_1x_city_20190727-9b3c56a5.pth) |
|    R-50-FPN     | paper   |   1x    | 800-1024 | Backbone    | -  | - | - | - |  31.5   | - |

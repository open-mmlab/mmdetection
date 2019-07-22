# Benchmark and Model Zoo

## Environment

### Hardware

- 1 NVIDIA GeForce GTX 1080 Ti GPU

## Common settings

- All baselines were trained using 1 GPU with a batch size of 2 (2 images per GPU).
- All models were trained on `cityscapes_train`, and tested on `cityscapes_val`.
- 1x training schedule indicates 64 epochs which corresponds to slightly less than the 24k iterations reported in the original schedule from the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- All pytorch-style pretrained backbones on ImageNet are from PyTorch model zoo.


## Baselines

Download links and more models with different backbones and training schemes will be added to the model zoo. 


### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Scale | Pretraining | box AP | Download |
| :-------------: | :-----: | :-----: | :---: | :---------: | :----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | Backbone |  36.4  |          |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Scale | Pretraining | box AP | mask AP | Download |
| :-------------: | :-----: | :-----: | :---: | :---------: | :----: | :-----: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 | Backbone |  37.5  |  32.7   |          |
|    R-50-FPN     | paper   |   1x    | 800-1024 | Backbone |  -     |  31.5   |          |

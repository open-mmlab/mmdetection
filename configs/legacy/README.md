# Legacy Configs in MMDetection V1.x

Configs in this directory implements the legacy configs used by MMDetection V1.x and its model zoos.

To help users convert their models from V1.x to MMDetection V2.0, we provide v1.x configs to inference the converted v1.x models.
Due to the BC-breaking changes in MMDetection V2.0 from MMDetection V1.x, running inference with the same model weights in these two version will produce different results. The difference will cause within 1% AP absolute difference as can be found in the following table.

## Usage

To upgrade the model version, the users need to do the following steps.

### 1. Convert model weights

```bash
python -u tools/upgrade_model_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH} --num-classes ${NUM_CLASSES}

```
- OLD_MODEL_PATH: the path to load the model weights in 1.x version.
- NEW_MODEL_PATH: the path to save the converted model weights in 2.0 version.
- NUM_CLASSES: number of classes of the original model weights. Usually it is 81 for COCO dataset, 21 for VOC dataset.
    The number of classes in V2.0 models should equal to the number of classes in V1.x models + 1.

### 2. Use the legacy configs

After converting the model weights, create a config that uses the legacy settings, e.g. `LegacyAnchorGenerator` and `RoIAlignV1`. Then use the config to test the model weights. The obtained results should be close to that in V1.x.

## Performance

|    Backbone     |  Style  | Lr schd | V1.x box AP | V1.x mask AP | Download | V2.0 box AP | V2.0 mask AP |
| :-------------: | :-----: | :-----: | :------:| :-----: | :------------------------------------------------------------------------------------------------------------------------------: | :------:| :-----: | 
|    [R-50-FPN](./mask_rcnn_r50_fpn_1x_coco_v1.py)     | pytorch |   1x    |  37.3  |  34.2   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth)| 36.4 | 33.7

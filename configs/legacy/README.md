# Legacy Configs in MMDetection V1.x

Configs in this directory implements the legacy configs used by MMDetection V1.x and its model zoos.

To help users convert their models from V1.x to MMDetection V2.0, we provide v1.x configs to inference the converted v1.x models.
Due to the BC-breaking changes in MMDetection V2.0 from MMDetection V1.x, running inference with the same model weights in these two version will produce different results. The difference will cause within 1% AP absolute difference as can be found in the following table.

## Usage

To upgrade the model version, the users need to do the following steps.

### 1. Convert model weights
Since all the detector's classification is reordered, all the legacy model weights need to go through the conversion process.
For two-stage detectors, their wegihts need to be upgraded since MMDetection V2.0 refactors all the two-stage detectors with `RoIHead`.
For example, to convert Mask R-CNN and Faster R-CNN, their weights need to be converted as the following

```bash
python -u tools/upgrade_model_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH} --num-classes ${NUM_CLASSES} [--upgrade-retina] [--ssd] [--reg-cls-agnostic]

```
- OLD_MODEL_PATH: the path to load the model weights in 1.x version.
- NEW_MODEL_PATH: the path to save the converted model weights in 2.0 version.
- NUM_CLASSES: number of classes of the original model weights. Usually it is 81 for COCO dataset, 21 for VOC dataset.
The number of classes in V2.0 models should equal to the number of classes in V1.x models + 1.

Optional arguments:
- `--upgrade-retina` Option for some RetinaNet models trained in older codebase (lower than V1.0), which will also upgrade the model weights' name.
- `--ssd` Option for the SSD Detector, the SSD Detector only need to reorder its classification branch.
- `--reg-cls-agnostic` Option for Cascade methods whose regression branch is class agnostic.

### 2. Use configs with legacy settings

After converting the model weights, create a config that uses the legacy settings, e.g. `LegacyAnchorGenerator`, `LegacyDeltaXYWHBBoxCoder` and `RoIAlign` without aliggment. For models using ResNet Caffe backbones, they also need to change the pretrain name and the corresponding `img_norm_cfg`.
An example is in [`retinanet_r50_caffe_fpn_1x_coco_v1.py`](retinanet_r50_caffe_fpn_1x_coco_v1.py)
Then use the config to test the model weights. The obtained results should be close to that in V1.x.

## Performance

Some configs in this directory have been tested as the following
|    Method    |  Style  | Lr schd | V1.x box AP | V1.x mask AP | V2.0 box AP | V2.0 mask AP |Download |
| :-------------: | :-----: | :-----: | :------:| :-----: |:------:| :-----: |:------------------------------------------------------------------------------------------------------------------------------: |
|[Mask R-CNN R-50-FPN](./mask_rcnn_r50_fpn_1x_coco_v1.py)     | pytorch |   1x    |  37.3  |  34.2   | 36.8 | 33.9 |[model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth)|
|[RetinaNet R-50-FPN](./retinanet_r50_caffe_fpn_1x_coco_v1.py)|  caffe  |   1x    |  35.8  | - | 35.4 | - |
|[RetinaNet R-50-FPN](./retinanet_r50_fpn_1x_coco_v1.py)| pytorch |   1x |  35.6 |-|35.2|   -|[model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth)     |
|[Cascade Mask R-CNN R-50-FPN](./cascade_mask_rcnn_r50_fpn_1x_coco_v1.py)     | pytorch |   1x    |  41.2  |  35.7   |40.8| 35.6|     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth)     |
|[SSD300-VGG16](./ssd300_coco_v1.py)  | caffe |  120e   | 25.7  |-|25.4|-| [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth) |

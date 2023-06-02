# Legacy Configs in MMDetection V1.x

<!-- [OTHERS] -->

Configs in this directory implement the legacy configs used by MMDetection V1.x and its model zoos.

To help users convert their models from V1.x to MMDetection V2.0, we provide v1.x configs to inference the converted v1.x models.
Due to the BC-breaking changes in MMDetection V2.0 from MMDetection V1.x, running inference with the same model weights in these two version will produce different results. The difference will cause within 1% AP absolute difference as can be found in the following table.

## Usage

To upgrade the model version, the users need to do the following steps.

### 1. Convert model weights

There are three main difference in the model weights between V1.x and V2.0 codebases.

1. Since the class order in all the detector's classification branch is reordered, all the legacy model weights need to go through the conversion process.
2. The regression and segmentation head no longer contain the background channel. Weights in these background channels should be removed to fix in the current codebase.
3. For two-stage detectors, their wegihts need to be upgraded since MMDetection V2.0 refactors all the two-stage detectors with `RoIHead`.

The users can do the same modification as mentioned above for the self-implemented
detectors. We provide a scripts `tools/model_converters/upgrade_model_version.py` to convert the model weights in the V1.x model zoo.

```bash
python tools/model_converters/upgrade_model_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH} --num-classes ${NUM_CLASSES}

```

- OLD_MODEL_PATH: the path to load the model weights in 1.x version.
- NEW_MODEL_PATH: the path to save the converted model weights in 2.0 version.
- NUM_CLASSES: number of classes of the original model weights. Usually it is 81 for COCO dataset, 21 for VOC dataset.
  The number of classes in V2.0 models should be equal to that in V1.x models - 1.

### 2. Use configs with legacy settings

After converting the model weights, checkout to the v1.2 release to find the corresponding config file that uses the legacy settings.
The V1.x models usually need these three legacy modules: `LegacyAnchorGenerator`, `LegacyDeltaXYWHBBoxCoder`, and `RoIAlign(align=False)`.
For models using ResNet Caffe backbones, they also need to change the pretrain name and the corresponding `img_norm_cfg`.
An example is in [`retinanet_r50-caffe_fpn_1x_coco_v1.py`](retinanet_r50-caffe_fpn_1x_coco_v1.py)
Then use the config to test the model weights. For most models, the obtained results should be close to that in V1.x.
We provide configs of some common structures in this directory.

## Performance

The performance change after converting the models in this directory are listed as the following.

|           Method            |  Style  | Lr schd | V1.x box AP | V1.x mask AP | V2.0 box AP | V2.0 mask AP |                       Config                        |                                                             Download                                                              |
| :-------------------------: | :-----: | :-----: | :---------: | :----------: | :---------: | :----------: | :-------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
|     Mask R-CNN R-50-FPN     | pytorch |   1x    |    37.3     |     34.2     |    36.8     |     33.9     |     [config](./mask-rcnn_r50_fpn_1x_coco_v1.py)     |     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth)     |
|     RetinaNet R-50-FPN      |  caffe  |   1x    |    35.8     |      -       |    35.4     |      -       |  [config](./retinanet_r50-caffe_fpn_1x_coco_v1.py)  |                                                                                                                                   |
|     RetinaNet R-50-FPN      | pytorch |   1x    |    35.6     |      -       |    35.2     |      -       |     [config](./retinanet_r50_fpn_1x_coco_v1.py)     |     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth)     |
| Cascade Mask R-CNN R-50-FPN | pytorch |   1x    |    41.2     |     35.7     |    40.8     |     35.6     | [config](./cascade-mask-rcnn_r50_fpn_1x_coco_v1.py) | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth) |
|        SSD300-VGG16         |  caffe  |  120e   |    25.7     |      -       |    25.4     |      -       |            [config](./ssd300_coco_v1.py)            | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth) |

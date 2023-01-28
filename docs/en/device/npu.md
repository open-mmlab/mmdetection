# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV on NPU devices

Here we use 8 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/ssd/ssd300_coco.py 8
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/ssd/ssd300_coco.py
```

## Models Results

|        Model         | box AP | mask AP | Config                                                                                                                        | Download                                                                                                     |
| :------------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
|     [ssd300](<>)     |  25.6  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssd300_fp16_coco.py)                               | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/ssd300_coco.log.json)                              |
|     [ssd512](<>)     |  29.4  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssd512_fp16_coco.py)                               | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/ssd512_coco.log.json)                              |
| [ssdlite-mbv2\*](<>) |  20.2  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py)          | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/ssdlite_mobilenetv2_scratch_600e_coco.log.json)    |
| [retinanet-r18](<>)  |  31.8  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r18_fpn_1x8_1x_coco.py)            | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/retinanet_r18_fpn_1x8_1x_coco.log.json)            |
| [retinanet-r50](<>)  |  36.6  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_fpn_fp16_1x_coco.py)           | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/retinanet_r50_fpn_1x_coco.log.json)                |
|   [yolov3-608](<>)   |  34.7  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py)         | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/yolov3_d53_fp16_mstrain-608_273e_coco.log.json)    |
|  [yolox-s\*\*](<>)   |  39.9  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/yolox_s_8x8_300e_coco.py)                        | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/yolox_s_8x8_300e_coco.log.json)                    |
| [centernet-r18](<>)  |  26.1  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/centernet/centernet_resnet18_140e_cocoo.py)            | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/centernet_resnet18_140e_coco.log.jsonn)            |
|   [fcos-r50\*](<>)   |  36.1  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_fp16_1x_bs8x8_coco.py) | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/fcos_r50_caffe_fpn_gn-head_1x_coco_bs8x8.log.json) |
|   [solov2-r50](<>)   |  ---   |  34.7   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/solov2/solov2_r50_fpn_1x_coco.py)                      | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/solov2_r50_fpn_1x_coco.log.json)                   |

**Notes:**

- If not specially marked, the results on NPU are the same as those on the GPU with FP32.
- (\*) The results on the NPU of these models are aligned with the results of the mixed-precision training on the GPU,
  but are lower than the results of the FP32. This situation is mainly related to the phase of the model itself in
  mixed-precision training, users may need to adjust the hyperparameters to achieve better results.
- (\*\*) The accuracy of yolox-s on the GPU in mixed precision is 40.1, with `persister_woker=True` in the data loader config by default.
  There are currently some bugs on NPUs that prevent the last few epochs from running, but the accuracy is less affected and the difference can be ignored.

## High-performance Model Result on Ascend Device

Introduction to optimization:

1. Modify the loop calculation as a whole batch calculation to reduce the number of instructions issued.
2. Modify the index calculation to mask calculation, because the SIMD architecture is good at processing continuous data calculation.

|           Model            |                                                          Config                                                           | v100 iter time |       910A iter time       |
| :------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :------------: | :------------------------: |
|    [ascend-ssd300](<>)     |          [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ascend_ssd300_fp16_coco.py)           |  0.165s/iter   | 0.383s/iter -> 0.13s/iter  |
| [ascend-retinanet-r18](<>) | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/ascend_retinanet_r18_fpn_1x8_1x_coco.py) |  0.567s/iter   | 0.780s/iter -> 0.420s/iter |

**All above models are provided by Huawei Ascend group.**

# NPU (HUAWEI Ascend)

## Usage

Please refer to [link](https://github.com/open-mmlab/mmcv/blob/master/docs/zh_cn/get_started/build.md) installing mmcv
on NPU Devices.

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
| [\*ssdlite-mbv2](<>) |  20.2  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py)          | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/ssdlite_mobilenetv2_scratch_600e_coco.log.json)    |
| [retinanet-r18](<>)  |  31.8  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r18_fpn_1x8_1x_coco.py)            | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/retinanet_r18_fpn_1x8_1x_coco.log.json)            |
| [retinanet-r50](<>)  |  36.6  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_fpn_fp16_1x_coco.py)           | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/retinanet_r50_fpn_1x_coco.log.json)                |
|   [yolov3-608](<>)   |  34.7  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolo/yolov3_d53_fp16_mstrain-608_273e_coco.py)         | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/yolov3_d53_fp16_mstrain-608_273e_coco.log.json)    |
|  [\*\*yolox-s](<>)   |  39.9  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/yolox_s_8x8_300e_coco.py)                        | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/yolox_s_8x8_300e_coco.log.json)                    |
| [centernet-r18](<>)  |  26.1  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/centernet/centernet_resnet18_140e_cocoo.py)            | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/centernet_resnet18_140e_coco.log.jsonn)            |
|   [\*fcos-r50](<>)   |  36.1  |   ---   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_fp16_1x_bs8x8_coco.py) | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/fcos_r50_caffe_fpn_gn-head_1x_coco_bs8x8.log.json) |
|   [solov2-r50](<>)   |  ---   |  34.7   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/solov2/solov2_r50_fpn_1x_coco.py)                      | [log](https://download.openmmlab.com/mmdetection/v2.0/npu/solov2_r50_fpn_1x_coco.log.json)                   |

**Notes:**

- If not specially marked, the results on NPU are the same as those on the GPU with FP32.
- (\*) The results on the NPU of these models are aligned with the results of the mixed-precision training on the GPU,
  but are lower than the results of the FP32. This situation is mainly related to the phase of the model itself in
  mixed-precision training, users please adjust the hyperparameters to achieve the best result by self.
- (\*\*) The accuracy of yolox-s on the GPU in mixed precision is 40.1; yolox-s is persister_woker enabled by default,
  and there are currently some bugs on NPUs that prevent the last few epochs from running, but the accuracy is less
  affected and can be ignored.

**All above models are provided by Huawei Ascend group.**

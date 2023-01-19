# NPU (华为 昇腾)

## 使用方法

请参考 [MMCV 的安装文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) 来安装 NPU 版本的 MMCV。

以下展示单机八卡场景的运行指令:

```shell
bash tools/dist_train.sh configs/ssd/ssd300_coco.py 8
```

以下展示单机单卡下的运行指令:

```shell
python tools/train.py configs/ssd/ssd300_coco.py
```

## 模型验证结果

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

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果结果相同。
- (\*) 这些模型在 NPU 上的结果与 GPU 上的混合精度训练结果一致，但低于 FP32 的结果。这种情况主要与模型本身在混合精度训练中的特点有关，
  用户可以自行调整超参数来获得更高精度。
- (\*\*) GPU 上 yolox-s 在混合精度下的精度为 40.1 低于readme中 40.5 的水平;默认情况下，yolox-s 启用 `persister_woker=True`，但这个参数
  目前在NPU上存在一些bug，会导致在最后几个epoch由于资源耗尽报错退出，对整体精度影响有限可以忽略。

## Ascend加速模块验证结果

优化方案简介：

1. 修改循环计算为一次整体计算，目的是减少下发指令数量。
2. 修改索引计算为掩码计算，原因是SIMD架构芯片擅长处理连续数据计算。

|           Model            |                                                          Config                                                           | v100 iter time |       910A iter time       |
| :------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :------------: | :------------------------: |
|    [ascend-ssd300](<>)     |          [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ssd/ascend_ssd300_fp16_coco.py)           |  0.165s/iter   | 0.383s/iter -> 0.13s/iter  |
| [ascend-retinanet-r18](<>) | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/ascend_retinanet_r18_fpn_1x8_1x_coco.py) |  0.567s/iter   | 0.780s/iter -> 0.420s/iter |

**以上模型结果由华为昇腾团队提供**

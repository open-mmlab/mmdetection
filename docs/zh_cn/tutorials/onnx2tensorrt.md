# 教程 9: ONNX 到 TensorRT 的模型转换（实验性支持）

> ## [尝试使用新的 MMDeploy 来部署你的模型](https://mmdeploy.readthedocs.io/)

<!-- TOC -->

- [教程 9: ONNX 到 TensorRT 的模型转换（实验性支持）](#%E6%95%99%E7%A8%8B-9-onnx-%E5%88%B0-tensorrt-%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E5%AE%9E%E9%AA%8C%E6%80%A7%E6%94%AF%E6%8C%81)
  - [如何将模型从 ONNX 转换为 TensorRT](#%E5%A6%82%E4%BD%95%E5%B0%86%E6%A8%A1%E5%9E%8B%E4%BB%8E-onnx-%E8%BD%AC%E6%8D%A2%E4%B8%BA-tensorrt)
    - [先决条件](#%E5%85%88%E5%86%B3%E6%9D%A1%E4%BB%B6)
    - [用法](#%E7%94%A8%E6%B3%95)
  - [如何评估导出的模型](#%E5%A6%82%E4%BD%95%E8%AF%84%E4%BC%B0%E5%AF%BC%E5%87%BA%E7%9A%84%E6%A8%A1%E5%9E%8B)
  - [支持转换为 TensorRT 的模型列表](#%E6%94%AF%E6%8C%81%E8%BD%AC%E6%8D%A2%E4%B8%BA-tensorrt-%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)
  - [提醒](#%E6%8F%90%E9%86%92)
  - [常见问题](#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)

<!-- TOC -->

## 如何将模型从 ONNX 转换为 TensorRT

### 先决条件

1. 请参考 [get_started.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) 从源码安装 MMCV 和 MMDetection。
2. 请参考 [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_op.html) 和 [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/en/deployment/tensorrt_plugin.md/) 安装支持 ONNXRuntime 自定义操作和 TensorRT 插件的 `mmcv-full`。
3. 使用工具 [pytorch2onnx](https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html) 将模型从 PyTorch 转换为 ONNX。

### 用法

```bash
python tools/deployment/onnx2tensorrt.py \
    ${CONFIG} \
    ${MODEL} \
    --trt-file ${TRT_FILE} \
    --input-img ${INPUT_IMAGE_PATH} \
    --shape ${INPUT_IMAGE_SHAPE} \
    --min-shape ${MIN_IMAGE_SHAPE} \
    --max-shape ${MAX_IMAGE_SHAPE} \
    --workspace-size {WORKSPACE_SIZE} \
    --show \
    --verify \
```

所有参数的说明：

- `config`: 模型配置文件的路径。
- `model`: ONNX 模型文件的路径。
- `--trt-file`: 输出 TensorRT 引擎文件的路径。如果未指定，它将被设置为 `tmp.trt`。
- `--input-img`: 用于追踪和转换的输入图像的路径。默认情况下，它将设置为 `demo/demo.jpg`。
- `--shape`: 模型输入的高度和宽度。如果未指定，它将设置为 `400 600`。
- `--min-shape`: 模型输入的最小高度和宽度。如果未指定，它将被设置为与 `--shape` 相同。
- `--max-shape`: 模型输入的最大高度和宽度。如果未指定，它将被设置为与 `--shape` 相同。
- `--workspace-size`: 构建 TensorRT 引擎所需的 GPU 工作空间大小（以 GiB 为单位）。如果未指定，它将设置为 `1` GiB。
- `--show`: 确定是否显示模型的输出。如果未指定，它将设置为 `False`。
- `--verify`: 确定是否在 ONNXRuntime 和 TensorRT 之间验证模型的正确性。如果未指定，它将设置为 `False`。
- `--verbose`: 确定是否打印日志消息。它对调试很有用。如果未指定，它将设置为 `False`。

例子:

```bash
python tools/deployment/onnx2tensorrt.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    checkpoints/retinanet_r50_fpn_1x_coco.onnx \
    --trt-file checkpoints/retinanet_r50_fpn_1x_coco.trt \
    --input-img demo/demo.jpg \
    --shape 400 600 \
    --show \
    --verify \
```

## 如何评估导出的模型

我们准备了一个工具 `tools/deplopyment/test.py` 来评估 TensorRT 模型。

请参阅以下链接以获取更多信息。

- [如何评估导出的模型](pytorch2onnx.md#how-to-evaluate-the-exported-models)
- [结果和模型](pytorch2onnx.md#results-and-models)

## 支持转换为 TensorRT 的模型列表

下表列出了确定可转换为 TensorRT 的模型。

|       Model        |                              Config                              | Dynamic Shape | Batch Inference | Note |
| :----------------: | :--------------------------------------------------------------: | :-----------: | :-------------: | :--: |
|        SSD         |                   `configs/ssd/ssd300_coco.py`                   |       Y       |        Y        |      |
|        FSAF        |              `configs/fsaf/fsaf_r50_fpn_1x_coco.py`              |       Y       |        Y        |      |
|        FCOS        |         `configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py`         |       Y       |        Y        |      |
|       YOLOv3       |        `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`        |       Y       |        Y        |      |
|     RetinaNet      |         `configs/retinanet/retinanet_r50_fpn_1x_coco.py`         |       Y       |        Y        |      |
|    Faster R-CNN    |       `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`       |       Y       |        Y        |      |
|   Cascade R-CNN    |      `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`      |       Y       |        Y        |      |
|     Mask R-CNN     |         `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`         |       Y       |        Y        |      |
| Cascade Mask R-CNN |   `configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py`    |       Y       |        Y        |      |
|     PointRend      | `configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py` |       Y       |        Y        |      |

注意:

- *以上所有模型通过 Pytorch==1.6.0, onnx==1.7.0 与 TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0 测试*

## 提醒

- 如果您在上面列出的模型中遇到任何问题，请创建 issue，我们会尽快处理。对于未包含在列表中的模型，由于资源有限，我们可能无法在此提供太多帮助。请尝试深入挖掘并自行调试。
- 由于此功能是实验性的，并且可能会快速更改，因此请始终尝试使用最新的 `mmcv` 和 `mmdetecion`。

## 常见问题

- 空

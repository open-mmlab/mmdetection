# Tutorial 9: ONNX to TensorRT (Experimental)

<!-- TOC -->

- [Tutorial 9: ONNX to TensorRT (Experimental)](#tutorial-9-onnx-to-tensorrt-experimental)
  - [How to convert models from ONNX to TensorRT](#how-to-convert-models-from-onnx-to-tensorrt)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [How to evaluate the exported models](#how-to-evaluate-the-exported-models)
  - [List of supported models convertable to TensorRT](#list-of-supported-models-convertable-to-tensorrt)
  - [Reminders](#reminders)
  - [FAQs](#faqs)

<!-- TOC -->

## How to convert models from ONNX to TensorRT

### Prerequisite

1. Please refer to [get_started.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation of MMCV and MMDetection from source.
2. Please refer to [ONNXRuntime in mmcv](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) and [TensorRT plugin in mmcv](https://github.com/open-mmlab/mmcv/blob/master/docs/tensorrt_plugin.md/) to install `mmcv-full` with ONNXRuntime custom ops and TensorRT plugins.
3. Use our tool [pytorch2onnx](https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html) to convert the model from PyTorch to ONNX.

### Usage

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

Description of all arguments:

- `config` : The path of a model config file.
- `model` : The path of an ONNX model file.
- `--trt-file`: The Path of output TensorRT engine file. If not specified, it will be set to `tmp.trt`.
- `--input-img` : The path of an input image for tracing and conversion. By default, it will be set to `demo/demo.jpg`.
- `--shape`: The height and width of model input. If not specified, it will be set to `400 600`.
- `--min-shape`: The minimum height and width of model input. If not specified, it will be set to the same as `--shape`.
- `--max-shape`: The maximum height and width of model input. If not specified, it will be set to the same as `--shape`.
- `--workspace-size` : The required GPU workspace size in GiB to build TensorRT engine. If not specified, it will be set to `1` GiB.
- `--show`: Determines whether to show the outputs of the model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of models between ONNXRuntime and TensorRT. If not specified, it will be set to `False`.
- `--verbose`: Determines whether to print logging messages. It's useful for debugging. If not specified, it will be set to `False`.

Example:

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

## How to evaluate the exported models

We prepare a tool `tools/deplopyment/test.py` to evaluate TensorRT models.

Please refer to following links for more information.

- [how-to-evaluate-the-exported-models](pytorch2onnx.md#how-to-evaluate-the-exported-models)
- [results-and-models](pytorch2onnx.md#results-and-models)

## List of supported models convertable to TensorRT

The table below lists the models that are guaranteed to be convertable to TensorRT.

|    Model     |                        Config                        | Dynamic Shape | Batch Inference | Note  |
| :----------: | :--------------------------------------------------: | :-----------: | :-------------: | :---: |
|     SSD      |             `configs/ssd/ssd300_coco.py`             |       Y       |        Y        |       |
|     FSAF     |        `configs/fsaf/fsaf_r50_fpn_1x_coco.py`        |       Y       |        Y        |       |
|     FCOS     |   `configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py`   |       Y       |        Y        |       |
|    YOLOv3    |  `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`  |       Y       |        Y        |       |
|  RetinaNet   |   `configs/retinanet/retinanet_r50_fpn_1x_coco.py`   |       Y       |        Y        |       |
| Faster R-CNN | `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py` |       Y       |        Y        |       |
|  Mask R-CNN  |   `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`   |       Y       |        Y        |       |
|  PointRend   | `configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py` |   Y    |   Y        |       |

Notes:

- *All models above are tested with Pytorch==1.6.0, onnx==1.7.0 and TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0*

## Reminders

- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, we may not provide much help here due to the limited resources. Please try to dig a little deeper and debug by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmdetecion`.

## FAQs

- None

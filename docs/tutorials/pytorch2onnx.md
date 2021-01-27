# Tutorial 8: Pytorch to ONNX (Experimental)

<!-- TOC -->

- [Tutorial 8: Pytorch to ONNX (Experimental)](#tutorial-8-pytorch-to-onnx-experimental)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [List of supported models exportable to ONNX](#list-of-supported-models-exportable-to-onnx)
  - [Reminders](#reminders)
  - [FAQs](#faqs)

<!-- TOC -->

## How to convert models from Pytorch to ONNX

### Prerequisite

1. Please refer to [get_started.md](../get_started.md) for installation of MMCV and MMDetection.
2. Install onnx and onnxruntime

  ```shell
  pip install onnx onnxruntime
  ```

### Usage

```bash
python tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --input-img ${INPUT_IMAGE_PATH} \
    --shape ${IMAGE_SHAPE} \
    --mean ${IMAGE_MEAN} \
    --std ${IMAGE_STD} \
    --dataset ${DATASET_NAME} \
    --test-img ${TEST_IMAGE_PATH} \
    --opset-version ${OPSET_VERSION} \
    --show \
    --verify \
```

Description of all arguments:

- `config` : The path of a model config file.
- `checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--input-img` : The path of an input image for tracing and conversion. By default, it will be set to `tests/data/color.jpg`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `800 1216`.
- `--mean` : Three mean values for the input image. If not specified, it will be set to `123.675 116.28 103.53`.
- `--std` : Three std values for the input image. If not specified, it will be set to `58.395 57.12 57.375`.
- `--dataset` : The dataset name for the input model. If not specified, it will be set to `coco`.
- `--test-img` : The path of an image to verify the exported ONNX model. By default, it will be set to `None`, meaning it will use `--input-img` for verification.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.

Example:

```bash
python tools/deployment/pytorch2onnx.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    --output-file checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 608 608 \
    --mean 0 0 0 \
    --std 255 255 255 \
    --show \
    --verify \
```

## List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|    Model    |                        Config                        | Note  |
| :---------: | :--------------------------------------------------: | :---: |
|     SSD     |             `configs/ssd/ssd300_coco.py`             |       |
|   YOLOv3    |  `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`  |       |
|    FSAF     |        `configs/fsaf/fsaf_r50_fpn_1x_coco.py`        |       |
|  RetinaNet  |   `configs/retinanet/retinanet_r50_fpn_1x_coco.py`   |       |
| Faster-RCNN | `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py` |       |

Notes:

- *All models above are tested with Pytorch==1.6.0*

## Reminders

- When the input model has custom op such as `RoIAlign` and if you want to verify the exported ONNX model, you may have to build `mmcv` with [ONNXRuntime](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) from source.
- `mmcv.onnx.simplify` feature is based on [onnx-simplifier](https://github.com/daquexian/onnx-simplifier). If you want to try it, please refer to [onnx in `mmcv`](https://mmcv.readthedocs.io/en/latest/onnx.html) and [onnxruntime op in `mmcv`](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) for more information.
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmdetecion`.

## FAQs

- None

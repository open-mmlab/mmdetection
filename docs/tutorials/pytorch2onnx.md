# Model conversion from Pytroch to ONNX

<!-- TOC -->

- [Model conversion from Pytroch to ONNX](#model-conversion-from-pytroch-to-onnx)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
  - [List of supported models exportable to ONNX](#list-of-supported-models-exportable-to-onnx)
  - [Reminders](#reminders)
  - [FAQs](#faqs)

<!-- TOC -->

## How to convert models from Pytorch to ONNX

### Prerequisite

1. Please refer to [get_started.md](../../docs/get_started.md) for installation of MMCV and MMDetection.
2. Install onnx and onnxruntime

  ```shell
  pip install onnx onnxruntime
  ```

### Usage

Example of usage with command line:

```bash
python tools/pytorch2onnx.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    --output-file checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco-squeeze.onnx \
    --input-img demo/demo.jpg \
    --show \
    --verify \
    --view \
    --shape 608 608 \
    --mean 0 0 0 \
    --std 255 255 255 \
```

Key arguments:

- `config` : The path of a model config file.
- `checkpoints` : The path of a model checkpoint file.
- `--input-img` : The path of an input image for tracing.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `800 1216`.
- `--verify`: Determines whether to verify the correctness of a exported model. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model. If not specified, it will be set to `False`.
- `--output-file`: The output onnx model name. If not specified, it will be set to `tmp.onnx`.

For a more detailed description of other arguments, pls. use `python tools/pytorch2onnx.py --help` for help.

## List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.
|    Model    |                                           Config                                           | Note  |
| :---------: | :----------------------------------------------------------------------------------------: | :---: |
|     SSD     |                      [ssd300_coco](../../configs/ssd/ssd300_coco.py)                       |
|   YOLOv3    | [yolov3_d53_mstrain-608_273e_coco](../../configs/yolo/yolov3_d53_mstrain-608_273e_coco.py) |       |
|    FSAF     |             [fsaf_r50_fpn_1x_coco](../../configs/fsaf/fsaf_r50_fpn_1x_coco.py)             |       |
|  RetinaNet  |     [retinanet_r50_fpn_1x_coco](../../configs/retinanet/retinanet_r50_fpn_1x_coco.py)      |       |
| Faster-RCNN |  [faster_rcnn_r50_fpn_1x_coco](../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)   |       |

Notes:

- *All models above are tested with Pytorch==1.6.0*

## Reminders

- If you meet any problem with listed [models](#list) above, pls. create an issue and it would be taken care of soon. For models not included in the list, pls. try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.
- Because this feature is experimental and may change fast, pls. always try with latest `mmcv` and `mmdetecion`.

## FAQs

- None

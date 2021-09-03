# Tutorial 11: Pytorch to OpenVINO (Experimental)

<!-- TOC -->

- [Tutorial 11: Pytorch to OpenVINO (Experimental)](#tutorial-11-pytorch-to-openvino-experimental)
	- [How to convert models from Pytorch to OpenVINO](#how-to-convert-models-from-pytorch-to-openvino)
		- [Prerequisite](#prerequisite)
		- [Usage](#usage)
		- [Description of all arguments](#description-of-all-arguments)
	- [List of supported models exportable to OpenVINO](#list-of-supported-models-exportable-to-openvino)

<!-- TOC -->

## How to convert models from Pytorch to OpenVINO

### Prerequisite

1. Install the prerequisites following [get_started.md/Prepare environment](../get_started.md).
2. Build custom operators for ONNX Runtime and install MMCV manually following [How to build custom operators for ONNX Runtime](https://github.com/open-mmlab/mmcv/blob/master/docs/deployment/onnxruntime_op.md/#how-to-build-custom-operators-for-onnx-runtime)
3. Install MMdetection manually following steps 2-3 in [get_started.md/Install MMdetection](../get_started.md).
4. Install [OpenVINO](https://docs.openvinotoolkit.org/latest/installation_guides.html).

### Usage

```bash
python tools/deployment/pytorch2openvino.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${PYTORCH2ONNX_ARGS} \
    --not_strip_doc_string \
```

### Description of all arguments

- `config` : The path of a model config file.
- `checkpoint` : The path of a model checkpoint file.
- `pytorch2onnx_args`: Script arguments for export to ONNX. The list of possible values for `pytorch2onnx_args` can be found [here](pytorch2onnx.md#how-to-convert-models-from-pytorch-to-onnx).
- `--not_strip_doc_string`: If this argument is used, then fields with debug information will remain in the ONNX graph.

Example:

```bash
python tools/deployment/pytorch2openvino.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    --output-file checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.onnx \
    --shape 608 608 \
    --dynamic-export \
    --opset-version 11 \
    --not_strip_doc_string
```

## List of supported models exportable to OpenVINO

The table below lists the models that are guaranteed to be exportable to OpenVINO.

|    Model     |                               Config                                | Dynamic Shape |                                     Note                                      |
| :----------: | :-----------------------------------------------------------------: | :-----------: | :---------------------------------------------------------------------------: |
|     FCOS     |      `configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py`       |       Y       ||                                                                               |
|     FSAF     |               `configs/fsaf/fsaf_r50_fpn_1x_coco.py`                |       Y       ||                                                                               |
|  RetinaNet   |          `configs/retinanet/retinanet_r50_fpn_1x_coco.py`           |       Y       ||                                                                               |
|     SSD      |                    `configs/ssd/ssd300_coco.py`                     |       Y       ||                                                                               |
|    YOLOv3    |         `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`          |       Y       ||                                                                               |
| Faster R-CNN |        `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`         |       Y       ||                                                                               |
| Cascade R-CNN| `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py`              |       Y       ||                                                                               |
|  Mask R-CNN  |          `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`           |       Y       ||                                                                               |
| Cascade Mask R-CNN  |  `configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py` |       Y       ||                                                                               |

Notes:

- For faster work in OpenVINO in the Faster-RCNN, Mask-RCNN, Cascade-RCNN, Cascade-Mask-RCNN models,
the RoiAlign operation is replaced with the [ExperimentalDetectronROIFeatureExtractor](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_detection_ExperimentalDetectronROIFeatureExtractor_6.html) operation in the ONNX graph.

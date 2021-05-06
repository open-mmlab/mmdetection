# Tutorial 8: Pytorch to ONNX (Experimental)

<!-- TOC -->

- [Tutorial 8: Pytorch to ONNX (Experimental)](#tutorial-8-pytorch-to-onnx-experimental)
  - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
    - [Prerequisite](#prerequisite)
    - [Usage](#usage)
    - [Description of all arguments](#description-of-all-arguments)
  - [How to evaluate ONNX models with ONNX Runtime](#how-to-evaluate-onnx-models-with-onnx-runtime)
    - [Prerequisite](#prerequisite-1)
    - [Usage](#usage-1)
    - [Description of all arguments](#description-of-all-arguments-1)
    - [Results and Models](#results-and-models)
  - [List of supported models exportable to ONNX](#list-of-supported-models-exportable-to-onnx)
  - [The Parameters of Non-Maximum Suppression in ONNX Export](#the-parameters-of-non-maximum-suppression-in-onnx-export)
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
    --cfg-options ${CFG_OPTIONS}
    --dynamic-export \
    --show \
    --verify \
    --simplify \
```

### Description of all arguments

- `config` : The path of a model config file.
- `checkpoint` : The path of a model checkpoint file.
- `--output-file`: The path of output ONNX model. If not specified, it will be set to `tmp.onnx`.
- `--input-img`: The path of an input image for tracing and conversion. By default, it will be set to `tests/data/color.jpg`.
- `--shape`: The height and width of input tensor to the model. If not specified, it will be set to `800 1216`.
- `--mean` : Three mean values for the input image. If not specified, it will be set to `123.675 116.28 103.53`.
- `--std` : Three std values for the input image. If not specified, it will be set to `58.395 57.12 57.375`.
- `--dataset` : The dataset name for the input model. If not specified, it will be set to `coco`.
- `--test-img` : The path of an image to verify the exported ONNX model. By default, it will be set to `None`, meaning it will use `--input-img` for verification.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--dynamic-export`: Determines whether to export ONNX model with dynamic input and output shapes. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model and whether to show detection outputs when `--verify` is set to `True`. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.

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
    --dynamic-export \
    --cfg-options \
      model.test_cfg.nms_pre=200  \
      model.test_cfg.max_per_img=200  \
      model.test_cfg.deploy_nms_pre=300 \
```

## How to evaluate ONNX models with ONNX Runtime

We prepare a tool `tools/deplopyment/test.py` to evaluate ONNX models with ONNX Runtime backend.

### Prerequisite

- Install onnx and onnxruntime-gpu

  ```shell
  pip install onnx onnxruntime-gpu
  ```

### Usage

```bash
python tools/deployment/test.py \
    ${CONFIG_FILE} \
    ${ONNX_FILE} \
    --out ${OUTPUT_FILE} \
    --format-only ${FORMAT_ONLY} \
    --eval ${EVALUATION_METRICS} \
    --show-dir ${SHOW_DIRECTORY} \
    ----show-score-thr ${SHOW_SCORE_THRESHOLD} \
    ----cfg-options ${CFG_OPTIONS} \
    ----eval-options ${EVALUATION_OPTIONS} \
```

### Description of all arguments

- `config`: The path of a model config file.
- `model`: The path of a ONNX model file.
- `--out`: The path of output result file in pickle format.
- `--format-only` : Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server. If not specified, it will be set to `False`.
- `--eval`: Evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC.
- `--show-dir`: Directory where painted images will be saved
- `--show-score-thr`: Score threshold. Default is set to `0.3`.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.
- `--eval-options`: Custom options for evaluation, the key-value pair in `xxx=yyy` format will be kwargs for `dataset.evaluate()` function

### Results and Models

<table border="1" class="docutils">
	<tr>
	    <th align="center">Model</th>
	    <th align="center">Config</th>
	    <th align="center">Metric</th>
	    <th align="center">PyTorch</th>
	    <th align="center">ONNX Runtime</th>
	</tr >
  <tr >
	    <td align="center">FCOS</td>
	    <td align="center"><code>configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.6</td>
	    <td align="center">36.5</td>
	</tr>
  <tr >
	    <td align="center">FSAF</td>
	    <td align="center"><code>configs/fsaf/fsaf_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.0</td>
	    <td align="center">36.0</td>
	</tr>
  <tr >
	    <td align="center">RetinaNet</td>
	    <td align="center"><code>configs/retinanet/retinanet_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.5</td>
	    <td align="center">36.4</td>
	</tr>
	<tr >
	    <td align="center" align="center" >SSD</td>
	    <td align="center" align="center"><code>configs/ssd/ssd300_coco.py</code></td>
	    <td align="center" align="center">Box AP</td>
	    <td align="center" align="center">25.6</td>
	    <td align="center" align="center">25.6</td>
	</tr>
  <tr >
	    <td align="center">YOLOv3</td>
	    <td align="center"><code>configs/yolo/yolov3_d53_mstrain-608_273e_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">33.5</td>
	    <td align="center">33.5</td>
	</tr>
  <tr >
	    <td align="center">Faster R-CNN</td>
	    <td align="center"><code>configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">37.4</td>
	    <td align="center">37.4</td>
	</tr>
  <tr >
	    <td align="center" rowspan="2">Mask R-CNN</td>
	    <td align="center" rowspan="2"><code>configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">38.2</td>
	    <td align="center">38.1</td>
	</tr>
	<tr>
	    <td align="center">Mask AP</td>
	    <td align="center">34.7</td>
	    <td align="center">33.7</td>
	</tr>
</table>

Notes:

- All ONNX models are evaluated with dynamic shape on coco dataset and images are preprocessed according to the original config file.

- Mask AP of Mask R-CNN drops by 1% for ONNXRuntime. The main reason is that the predicted masks are directly interpolated to original image in PyTorch, while they are at first interpolated to the preprocessed input image of the model and then to original image in ONNXRuntime.

## List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|    Model     |                          Config                          | Dynamic Shape | Batch Inference | Note  |
| :----------: | :------------------------------------------------------: | :-----------: | :-------------: | :---: |
|     FCOS     | `configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py` |       Y       |        Y        |       |
|     FSAF     |          `configs/fsaf/fsaf_r50_fpn_1x_coco.py`          |       Y       |        Y        |       |
|  RetinaNet   |     `configs/retinanet/retinanet_r50_fpn_1x_coco.py`     |       Y       |        Y        |       |
|     SSD      |               `configs/ssd/ssd300_coco.py`               |       Y       |        Y        |       |
|    YOLOv3    |    `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`    |       Y       |        Y        |       |
| Faster R-CNN |   `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`   |       Y       |        Y        |       |
|  Mask R-CNN  |     `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`     |       Y       |        Y        |       |

Notes:

- *All models above are tested with Pytorch==1.6.0 and onnxruntime==1.5.1*

- If the deployed backend platform is TensorRT, please add environment variables before running the file:

  ```bash
  export ONNX_BACKEND=MMCVTensorRT
  ```

- If you want to use the `--dynamic-export` parameter in the TensorRT backend to export ONNX, please remove the `--simplify` parameter, and vice versa.

## The Parameters of Non-Maximum Suppression in ONNX Export

In the process of exporting the ONNX model, we set some parameters for the NMS op to control the number of output bounding boxes. The following will introduce the parameter setting of the NMS op in the supported models. You can set these parameters through `--cfg-options`.

- `nms_pre`: The number of boxes before NMS. The default setting is `1000`.

- `deploy_nms_pre`: The number of boxes before NMS when exporting to ONNX model. The default setting is `0`.

- `max_per_img`: The number of boxes to be kept after NMS. The default setting is `100`.

- `max_output_boxes_per_class`: Maximum number of output boxes per class of NMS. The default setting is `200`.

## Reminders

- When the input model has custom op such as `RoIAlign` and if you want to verify the exported ONNX model, you may have to build `mmcv` with [ONNXRuntime](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) from source.
- `mmcv.onnx.simplify` feature is based on [onnx-simplifier](https://github.com/daquexian/onnx-simplifier). If you want to try it, please refer to [onnx in `mmcv`](https://mmcv.readthedocs.io/en/latest/onnx.html) and [onnxruntime op in `mmcv`](https://mmcv.readthedocs.io/en/latest/onnxruntime_op.html) for more information.
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmdetecion`.

## FAQs

- None

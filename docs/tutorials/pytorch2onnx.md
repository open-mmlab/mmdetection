# Tutorial 8: Pytorch to ONNX (Experimental)

<!-- TOC -->

- [Tutorial 8: Pytorch to ONNX (Experimental)](#tutorial-8-pytorch-to-onnx-experimental)
	- [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
		- [Prerequisite](#prerequisite)
		- [Usage](#usage)
		- [Description of all arguments](#description-of-all-arguments)
	- [How to evaluate the exported models](#how-to-evaluate-the-exported-models)
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

1. Install the prerequisites following [get_started.md/Prepare environment](../get_started.md).
2. Build custom operators for ONNX Runtime and install MMCV manually following [How to build custom operators for ONNX Runtime](https://github.com/open-mmlab/mmcv/blob/master/docs/deployment/onnxruntime_op.md/#how-to-build-custom-operators-for-onnx-runtime)
3. Install MMdetection manually following steps 2-3 in [get_started.md/Install MMdetection](../get_started.md).

### Usage

```bash
python tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --input-img ${INPUT_IMAGE_PATH} \
    --shape ${IMAGE_SHAPE} \
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
- `--test-img` : The path of an image to verify the exported ONNX model. By default, it will be set to `None`, meaning it will use `--input-img` for verification.
- `--opset-version` : The opset version of ONNX. If not specified, it will be set to `11`.
- `--dynamic-export`: Determines whether to export ONNX model with dynamic input and output shapes. If not specified, it will be set to `False`.
- `--show`: Determines whether to print the architecture of the exported model and whether to show detection outputs when `--verify` is set to `True`. If not specified, it will be set to `False`.
- `--verify`: Determines whether to verify the correctness of an exported model. If not specified, it will be set to `False`.
- `--simplify`: Determines whether to simplify the exported ONNX model. If not specified, it will be set to `False`.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.
- `--skip-postprocess`: Determines whether export model without post process. If not specified, it will be set to `False`. Notice: This is an experimental option. Only work for some single stage models. Users need to implement the post-process by themselves. We do not guarantee the correctness of the exported model.

Example:

```bash
python tools/deployment/pytorch2onnx.py \
    configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.pth \
    --output-file checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 608 608 \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1 \
```

## How to evaluate the exported models

We prepare a tool `tools/deplopyment/test.py` to evaluate ONNX models with ONNXRuntime and TensorRT.

### Prerequisite

- Install onnx and onnxruntime (CPU version)

  ```shell
  pip install onnx onnxruntime==1.5.1
  ```
- If you want to run the model on GPU, please remove the CPU version before using the GPU version.

  ```shell
  pip uninstall onnxruntime
  pip install onnxruntime-gpu
  ```

  Note: onnxruntime-gpu is version-dependent on CUDA and CUDNN, please ensure that your
  environment meets the requirements.

- Build custom operators for ONNX Runtime following [How to build custom operators for ONNX Runtime](https://github.com/open-mmlab/mmcv/blob/master/docs/deployment/onnxruntime_op.md/#how-to-build-custom-operators-for-onnx-runtime)

- Install TensorRT by referring to [How to build TensorRT plugins in MMCV](https://mmcv.readthedocs.io/en/latest/deployment/tensorrt_plugin.html#how-to-build-tensorrt-plugins-in-mmcv) (optional)

### Usage

```bash
python tools/deployment/test.py \
    ${CONFIG_FILE} \
    ${MODEL_FILE} \
    --out ${OUTPUT_FILE} \
    --backend ${BACKEND} \
    --format-only ${FORMAT_ONLY} \
    --eval ${EVALUATION_METRICS} \
    --show-dir ${SHOW_DIRECTORY} \
    ----show-score-thr ${SHOW_SCORE_THRESHOLD} \
    ----cfg-options ${CFG_OPTIONS} \
    ----eval-options ${EVALUATION_OPTIONS} \
```

### Description of all arguments

- `config`: The path of a model config file.
- `model`: The path of an input model file.
- `--out`: The path of output result file in pickle format.
- `--backend`: Backend for input model to run and should be `onnxruntime` or `tensorrt`.
- `--format-only` : Format the output results without perform evaluation. It is useful when you want to format the result to a specific format and submit it to the test server. If not specified, it will be set to `False`.
- `--eval`: Evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC.
- `--show-dir`: Directory where painted images will be saved
- `--show-score-thr`: Score threshold. Default is set to `0.3`.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in `xxx=yyy` format will be merged into config file.
- `--eval-options`: Custom options for evaluation, the key-value pair in `xxx=yyy` format will be kwargs for `dataset.evaluate()` function

Notes:

- If the deployed backend platform is TensorRT, please add environment variables before running the file:

  ```bash
  export ONNX_BACKEND=MMCVTensorRT
  ```

- If you want to use the `--dynamic-export` parameter in the TensorRT backend to export ONNX, please remove the `--simplify` parameter, and vice versa.

### Results and Models

<table border="1" class="docutils">
	<tr>
	    <th align="center">Model</th>
	    <th align="center">Config</th>
	    <th align="center">Metric</th>
	    <th align="center">PyTorch</th>
	    <th align="center">ONNX Runtime</th>
	    <th align="center">TensorRT</th>
	</tr >
  <tr >
	    <td align="center">FCOS</td>
	    <td align="center"><code>configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.6</td>
	    <td align="center">36.5</td>
	    <td align="center">36.3</td>
	</tr>
  <tr >
	    <td align="center">FSAF</td>
	    <td align="center"><code>configs/fsaf/fsaf_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.0</td>
	    <td align="center">36.0</td>
	    <td align="center">35.9</td>
	</tr>
  <tr >
	    <td align="center">RetinaNet</td>
	    <td align="center"><code>configs/retinanet/retinanet_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">36.5</td>
	    <td align="center">36.4</td>
	    <td align="center">36.3</td>
	</tr>
	<tr >
	    <td align="center" align="center" >SSD</td>
	    <td align="center" align="center"><code>configs/ssd/ssd300_coco.py</code></td>
	    <td align="center" align="center">Box AP</td>
	    <td align="center" align="center">25.6</td>
	    <td align="center" align="center">25.6</td>
	    <td align="center" align="center">25.6</td>
	</tr>
  <tr >
	    <td align="center">YOLOv3</td>
	    <td align="center"><code>configs/yolo/yolov3_d53_mstrain-608_273e_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">33.5</td>
	    <td align="center">33.5</td>
	    <td align="center">33.5</td>
	</tr>
  <tr >
	    <td align="center">Faster R-CNN</td>
	    <td align="center"><code>configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">37.4</td>
	    <td align="center">37.4</td>
	    <td align="center">37.0</td>
	</tr>
  <tr >
	    <td align="center">Cascade R-CNN</td>
	    <td align="center"><code>configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">40.3</td>
	    <td align="center">40.3</td>
	    <td align="center">40.1</td>
	</tr>

  <tr >
	    <td align="center" rowspan="2">Mask R-CNN</td>
	    <td align="center" rowspan="2"><code>configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">38.2</td>
	    <td align="center">38.1</td>
	    <td align="center">37.7</td>
	</tr>
	<tr>
	    <td align="center">Mask AP</td>
	    <td align="center">34.7</td>
	    <td align="center">33.7</td>
	    <td align="center">33.3</td>
	</tr>
  <tr >
	    <td align="center" rowspan="2">Cascade Mask R-CNN</td>
	    <td align="center" rowspan="2"><code>configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">41.2</td>
	    <td align="center">41.2</td>
	    <td align="center">40.9</td>
	</tr>
	<tr>
	    <td align="center">Mask AP</td>
	    <td align="center">35.9</td>
	    <td align="center">34.8</td>
	    <td align="center">34.5</td>
	</tr>


  <tr >
	    <td align="center">CornerNet</td>
	    <td align="center"><code>configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">40.6</td>
	    <td align="center">40.4</td>
		<td align="center">-</td>
	</tr>
  <tr >
	    <td align="center">DETR</td>
	    <td align="center"><code>configs/detr/detr_r50_8x2_150e_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">40.1</td>
	    <td align="center">40.1</td>
		<td align="center">-</td>
  </tr>
  <tr >
	    <td align="center" rowspan="2">PointRend</td>
	    <td align="center" rowspan="2"><code>configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py</code></td>
	    <td align="center">Box AP</td>
	    <td align="center">38.4</td>
	    <td align="center">38.4</td>
	    <td align="center">-</td>
  </tr>
  <tr>
	    <td align="center">Mask AP</td>
	    <td align="center">36.3</td>
	    <td align="center">35.2</td>
	    <td align="center">-</td>
  </tr>
</table>

Notes:

- All ONNX models are evaluated with dynamic shape on coco dataset and images are preprocessed according to the original config file. Note that CornerNet is evaluated without test-time flip, since currently only single-scale evaluation is supported with ONNX Runtime.

- Mask AP of Mask R-CNN drops by 1% for ONNXRuntime. The main reason is that the predicted masks are directly interpolated to original image in PyTorch, while they are at first interpolated to the preprocessed input image of the model and then to original image in other backend.

## List of supported models exportable to ONNX

The table below lists the models that are guaranteed to be exportable to ONNX and runnable in ONNX Runtime.

|    Model     |                               Config                                | Dynamic Shape | Batch Inference |                                     Note                                      |
| :----------: | :-----------------------------------------------------------------: | :-----------: | :-------------: | :---------------------------------------------------------------------------: |
|     FCOS     |      `configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py`       |       Y       |        Y        |                                                                               |
|     FSAF     |               `configs/fsaf/fsaf_r50_fpn_1x_coco.py`                |       Y       |        Y        |                                                                               |
|  RetinaNet   |          `configs/retinanet/retinanet_r50_fpn_1x_coco.py`           |       Y       |        Y        |                                                                               |
|     SSD      |                    `configs/ssd/ssd300_coco.py`                     |       Y       |        Y        |                                                                               |
|    YOLOv3    |         `configs/yolo/yolov3_d53_mstrain-608_273e_coco.py`          |       Y       |        Y        |                                                                               |
| Faster R-CNN |        `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py`         |       Y       |        Y        |                                                                               |
| Cascade R-CNN| `configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py` |   Y    |   Y        |       |
|  Mask R-CNN  |          `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py`           |       Y       |        Y        |                                                                               |
| Cascade Mask R-CNN  |   `configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py`   |       Y       |        Y        |       |
|  CornerNet   | `configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py` |       Y       |        N        | no flip, no batch inference, tested with torch==1.7.0 and onnxruntime==1.5.1. |
|     DETR     |                   `configs/detr/detr_r50_8x2_150e_coco.py`          |       Y       |        Y        | batch inference is *not recommended*                                          |
|  PointRend   | `configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py`    |       Y       |        Y        |                                                                               |

Notes:

- Minimum required version of MMCV is `1.3.5`

- *All models above are tested with Pytorch==1.6.0 and onnxruntime==1.5.1*, except for CornerNet. For more details about the
torch version when exporting CornerNet to ONNX, which involves `mmcv::cummax`, please refer to the [Known Issues](https://github.com/open-mmlab/mmcv/blob/master/docs/deployment/onnxruntime_op.md#known-issues) in mmcv.

- Though supported, it is *not recommended* to use batch inference in onnxruntime for `DETR`, because there is huge performance gap between ONNX and torch model (e.g. 33.5 vs 39.9 mAP on COCO for onnxruntime and torch respectively, with a batch size 2). The main reason for the gap is that these is non-negligible effect on the predicted regressions during batch inference for ONNX, since the predicted coordinates is normalized by `img_shape` (without padding) and should be converted to absolute format, but `img_shape` is not dynamically traceable thus the padded `img_shape_for_onnx` is used.

- Currently only single-scale evaluation is supported with ONNX Runtime, also `mmcv::SoftNonMaxSuppression` is only supported for single image by now.

## The Parameters of Non-Maximum Suppression in ONNX Export

In the process of exporting the ONNX model, we set some parameters for the NMS op to control the number of output bounding boxes. The following will introduce the parameter setting of the NMS op in the supported models. You can set these parameters through `--cfg-options`.

- `nms_pre`: The number of boxes before NMS. The default setting is `1000`.

- `deploy_nms_pre`: The number of boxes before NMS when exporting to ONNX model. The default setting is `0`.

- `max_per_img`: The number of boxes to be kept after NMS. The default setting is `100`.

- `max_output_boxes_per_class`: Maximum number of output boxes per class of NMS. The default setting is `200`.

## Reminders

- When the input model has custom op such as `RoIAlign` and if you want to verify the exported ONNX model, you may have to build `mmcv` with [ONNXRuntime](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_op.html) from source.
- `mmcv.onnx.simplify` feature is based on [onnx-simplifier](https://github.com/daquexian/onnx-simplifier). If you want to try it, please refer to [onnx in `mmcv`](https://mmcv.readthedocs.io/en/latest/deployment/onnx.html) and [onnxruntime op in `mmcv`](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_op.html) for more information.
- If you meet any problem with the listed models above, please create an issue and it would be taken care of soon. For models not included in the list, please try to dig a little deeper and debug a little bit more and hopefully solve them by yourself.
- Because this feature is experimental and may change fast, please always try with the latest `mmcv` and `mmdetecion`.

## FAQs

- None

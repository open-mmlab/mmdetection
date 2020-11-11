Apart from training/testing scripts, We provide lots of useful tools under the
 `tools/` directory.

## Log Analysis

`tools/analyze_logs.py` plots loss/mAP curves given a training
 log file. Run `pip install seaborn` first to install the dependency.

 ```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

![loss curve image](../resources/loss_curve.png)

Examples:

- Plot the classification loss of some run.

    ```shell
    python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

    ```shell
    python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
    ```

- Compare the bbox mAP of two runs in the same figure.

    ```shell
    python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
    ```

- Compute the average training speed.

    ```shell
    python tools/analyze_logs.py cal_train_time log.json [--include-outliers]
    ```

    The output is expected to be like the following.

    ```text
    -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
    slowest epoch 11, average time is 1.2024
    fastest epoch 1, average time is 1.1909
    time std over epochs is 0.0028
    average iter time: 1.1959 s/iter
    ```

## Visualization

### Visualize Datasets

`tools/browse_dataset.py` helps the user to browse a detection dataset (both
 images and bounding box annotations) visually, or save the image to a
  designated directory.

```shell
python tools/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

### Visualize Models

First, convert the model to ONNX as described
[here](#convert-mmdetection-model-to-onnx-experimental).
Note that currently only RetinaNet is supported, support for other models
 will be coming in later versions.
The converted model could be visualized by tools like [Netron](https://github.com/lutzroeder/netron).

### Visualize Predictions

If you need a lightweight GUI for visualizing the detection results, you can refer [DetVisGUI project](https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection).

## Error Analysis

`tools/coco_error_analysis.py` analyzes COCO results per category and by
 different criterion. It can also make a plot to provide useful
  information.

```shell
python tools/coco_error_analysis.py ${RESULT} ${OUT_DIR} [-h] [--ann ${ANN}] [--types ${TYPES[TYPES...]}]
```

## Model Complexity

`tools/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the results like this.

```text
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the
 number is absolutely correct. You may well use the result for simple
  comparisons, but double check it before you adopt it in technical reports or papers.

1. FLOPs are related to the input shape while parameters are not. The default
 input shape is (1, 3, 1280, 800).
2. Some operators are not counted into FLOPs like GN and custom operators
. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.
3. The FLOPs of two-stage detectors is dependent on the number of proposals.

## Model conversion

### MMDetection model to ONNX (experimental)

We provide a script to convert model to [ONNX](https://github.com/onnx/onnx) format. We also support comparing the output results between Pytorch and
 ONNX model for verification.

```shell
python tools/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --output_file ${ONNX_FILE} [--shape ${INPUT_SHAPE} --verify]
```

**Note**: This tool is still experimental. Some customized operators are not supported for now. We only support exporting RetinaNet model at this moment.

### MMDetection 1.x model to MMDetection 2.x

`tools/upgrade_model_version.py` upgrades a previous MMDetection checkpoint
 to the new version. Note that this script is not guaranteed to work as some
  breaking changes are introduced in the new version. It is recommended to
   directly use the new checkpoints.

```shell
python tools/upgrade_model_version.py ${IN_FILE} ${OUT_FILE} [-h] [--num-classes NUM_CLASSES]
```

### RegNet model to MMDetection

`tools/regnet2mmdet.py` convert keys in pycls pretrained RegNet models to
 MMDetection style.

```shell
python tools/regnet2mmdet.py ${SRC} ${DST} [-h]
```

### Detectron ResNet to Pytorch

`tools/detectron2pytorch.py` converts keys in the original detectron pretrained
 ResNet models to PyTorch style.

```shell
python tools/detectron2pytorch.py ${SRC} ${DST} ${DEPTH} [-h]
```

### Prepare a model for publishing

`tools/publish_model.py` helps users to prepare their model for publishing.

Before you upload a model to AWS, you may want to

1. convert model weights to CPU tensors
2. delete the optimizer states and
3. compute the hash of the checkpoint file and append the hash id to the
 filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

The final output filename will be `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

## Dataset Conversion

`tools/convert_datasets/` contains tools to convert the Cityscapes dataset
 and Pascal VOC dataset to the COCO format.

```shell
python tools/convert_datasets/cityscapes.py ${CITYSCAPES_PATH} [-h] [--img-dir ${IMG_DIR}] [--gt-dir ${GT_DIR}] [-o ${OUT_DIR}] [--nproc ${NPROC}]
python tools/convert_datasets/pascal_voc.py ${DEVKIT_PATH} [-h] [-o ${OUT_DIR}]
```

## Miscellaneous

### Evaluating a metric

`tools/eval_metric.py` evaluates certain metrics of a pkl result file
 according to a config file.

```shell
python tools/eval_metric.py ${CONFIG} ${PKL_RESULTS} [-h] [--format-only] [--eval ${EVAL[EVAL ...]}]
                      [--cfg-options ${CFG_OPTIONS [CFG_OPTIONS ...]}]
                      [--eval-options ${EVAL_OPTIONS [EVAL_OPTIONS ...]}]
```

### Print the entire config

`tools/print_config.py` prints the whole config verbatim, expanding all its
 imports.

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### Test the robustness of detectors

Please refer to [robustness_benchmarking.md](robustness_benchmarking.md).

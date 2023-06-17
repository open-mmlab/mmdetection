除了训练和测试脚本，我们还在 `tools/` 目录下提供了许多有用的工具。

## 日志分析

`tools/analysis_tools/analyze_logs.py` 可利用指定的训练 log 文件绘制 loss/mAP 曲线图，
第一次运行前请先运行 `pip install seaborn` 安装必要依赖.

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--eval-interval ${EVALUATION_INTERVAL}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

![loss curve image](../../../resources/loss_curve.png)

样例:

- 绘制分类损失曲线图

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
  ```

- 绘制分类损失、回归损失曲线图，保存图片为对应的 pdf 文件

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_bbox --out losses.pdf
  ```

- 在相同图像中比较两次运行结果的 bbox mAP

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
  ```

- 计算平均训练速度

  ```shell
  python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  输出以如下形式展示

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

## 结果分析

使用 `tools/analysis_tools/analyze_results.py` 可计算每个图像 mAP，随后根据真实标注框与预测框的比较结果，展示或保存最高与最低 top-k 得分的预测图像。

**使用方法**

```shell
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \
      ${PREDICTION_PATH} \
      ${SHOW_DIR} \
      [--show] \
      [--wait-time ${WAIT_TIME}] \
      [--topk ${TOPK}] \
      [--show-score-thr ${SHOW_SCORE_THR}] \
      [--cfg-options ${CFG_OPTIONS}]
```

各个参数选项的作用:

- `config`: model config 文件的路径。
- `prediction_path`:  使用 `tools/test.py` 输出的 pickle 格式结果文件。
- `show_dir`: 绘制真实标注框与预测框的图像存放目录。
- `--show`：决定是否展示绘制 box 后的图片，默认值为 `False`。
- `--wait-time`: show 时间的间隔，若为 0 表示持续显示。
- `--topk`: 根据最高或最低 `topk` 概率排序保存的图片数量，若不指定，默认设置为 `20`。
- `--show-score-thr`: 能够展示的概率阈值，默认为 `0`。
- `--cfg-options`: 如果指定，可根据指定键值对覆盖更新配置文件的对应选项

**样例**:
假设你已经通过 `tools/test.py` 得到了 pickle 格式的结果文件，路径为 './result.pkl'。

1. 测试 Faster R-CNN 并可视化结果，保存图片至 `results/`

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --show
```

2. 测试 Faster R-CNN 并指定 top-k 参数为 50，保存结果图片至 `results/`

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --topk 50
```

3. 如果你想过滤低概率的预测结果，指定 `show-score-thr` 参数

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --show-score-thr 0.3
```

## 可视化

### 可视化数据集

`tools/analysis_tools/browse_dataset.py` 可帮助使用者检查所使用的检测数据集（包括图像和标注），或保存图像至指定目录。

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

### 可视化模型

在可视化之前，需要先转换模型至 ONNX 格式，[可参考此处](#convert-mmdetection-model-to-onnx-experimental)。
注意，现在只支持 RetinaNet，之后的版本将会支持其他模型
转换后的模型可以被其他工具可视化[Netron](https://github.com/lutzroeder/netron)。

### 可视化预测结果

如果你想要一个轻量 GUI 可视化检测结果，你可以参考 [DetVisGUI project](https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection)。

## 误差分析

`tools/analysis_tools/coco_error_analysis.py` 使用不同标准分析每个类别的 COCO 评估结果。同时将一些有帮助的信息体现在图表上。

```shell
python tools/analysis_tools/coco_error_analysis.py ${RESULT} ${OUT_DIR} [-h] [--ann ${ANN}] [--types ${TYPES[TYPES...]}]
```

样例:

假设你已经把 [Mask R-CNN checkpoint file](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) 放置在文件夹 'checkpoint' 中（其他模型请在 [model zoo](./model_zoo.md) 中获取）。

为了保存 bbox 结果信息，我们需要用下列方式修改 `test_evaluator` :

1. 查找当前 config 文件相对应的  'configs/base/datasets' 数据集信息。
2. 用当前数据集 config 中的 test_evaluator 以及 test_dataloader 替换原始文件的 test_evaluator 以及 test_dataloader。
3. 使用以下命令得到 bbox 或 segmentation 的 json 格式文件。

```shell
python tools/test.py \
       configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
```

1. 得到每一类的 COCO bbox 误差结果，并保存分析结果图像至指定目录。（在 [config](../../../configs/_base_/datasets/coco_instance.py) 中默认目录是 './work_dirs/coco_instance/test')

```shell
python tools/analysis_tools/coco_error_analysis.py \
       results.bbox.json \
       results \
       --ann=data/coco/annotations/instances_val2017.json \
```

2. 得到每一类的 COCO 分割误差结果，并保存分析结果图像至指定目录。

```shell
python tools/analysis_tools/coco_error_analysis.py \
       results.segm.json \
       results \
       --ann=data/coco/annotations/instances_val2017.json \
       --types='segm'
```

## 模型服务部署

如果你想使用 [`TorchServe`](https://pytorch.org/serve/) 搭建一个 `MMDetection` 模型服务，可以参考以下步骤：

### 1. 安装 TorchServe

假设你已经成功安装了包含 `PyTorch` 和 `MMDetection` 的 `Python` 环境，那么你可以运行以下命令来安装 `TorchServe` 及其依赖项。有关更多其他安装选项，请参考[快速入门](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model)。

```shell
python -m pip install torchserve torch-model-archiver torch-workflow-archiver nvgpu
```

**注意**: 如果你想在 docker 中使用`TorchServe`，请参考[torchserve docker](https://github.com/pytorch/serve/blob/master/docker/README.md)。

### 2. 把 MMDetection 模型转换至 TorchServe

```shell
python tools/deployment/mmdet2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

### 3. 启动 `TorchServe`

```shell
torchserve --start --ncs \
  --model-store ${MODEL_STORE} \
  --models  ${MODEL_NAME}.mar
```

### 4. 测试部署效果

```shell
curl -O curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg
```

你可以得到下列 json 信息：

```json
[
  {
    "class_label": 16,
    "class_name": "dog",
    "bbox": [
      294.63409423828125,
      203.99111938476562,
      417.048583984375,
      281.62744140625
    ],
    "score": 0.9987992644309998
  },
  {
    "class_label": 16,
    "class_name": "dog",
    "bbox": [
      404.26019287109375,
      126.0080795288086,
      574.5091552734375,
      293.6662292480469
    ],
    "score": 0.9979367256164551
  },
  {
    "class_label": 16,
    "class_name": "dog",
    "bbox": [
      197.2144775390625,
      93.3067855834961,
      307.8505554199219,
      276.7560119628906
    ],
    "score": 0.993338406085968
  }
]
```

#### 结果对比

你也可以使用 `test_torchserver.py` 来比较 `TorchServe` 和 `PyTorch` 的结果，并可视化：

```shell
python tools/deployment/test_torchserver.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--device ${DEVICE}] [--score-thr ${SCORE_THR}] [--work-dir ${WORK_DIR}]
```

样例:

```shell
python tools/deployment/test_torchserver.py \
demo/demo.jpg \
configs/yolo/yolov3_d53_8xb8-320-273e_coco.py \
checkpoint/yolov3_d53_320_273e_coco-421362b6.pth \
yolov3 \
--work-dir ./work-dir
```

### 5. 停止 `TorchServe`

```shell
torchserve --stop
```

## 模型复杂度

`tools/analysis_tools/get_flops.py` 工具可用于计算指定模型的 FLOPs、参数量大小（改编自 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) ）。

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

获得的结果如下：

```text
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M
==============================
```

**注意**：这个工具还只是实验性质，我们不保证这个数值是绝对正确的。你可以将他用于简单的比较，但如果用于科技论文报告需要再三检查确认。

1. FLOPs 与输入的形状大小相关，参数量没有这个关系，默认的输入形状大小为 (1, 3, 1280, 800) 。
2. 一些算子并不计入 FLOPs，比如 GN 或其他自定义的算子。你可以参考 [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/utils/flops_counter.py) 查看更详细的说明。
3. 两阶段检测的 FLOPs 大小取决于 proposal 的数量。

## 模型转换

### MMDetection 模型转换至 ONNX 格式

我们提供了一个脚本用于转换模型至 [ONNX](https://github.com/onnx/onnx) 格式。同时还支持比较 Pytorch 与 ONNX 模型的输出结果以便对照。更详细的内容可以参考 [mmdeploy](https://github.com/open-mmlab/mmdeploy)。

### MMDetection 1.x 模型转换至 MMDetection 2.x 模型

`tools/model_converters/upgrade_model_version.py` 可将旧版本的 MMDetection checkpoints 转换至新版本。但要注意此脚本不保证在新版本加入非兼容更新后还能正常转换，建议您直接使用新版本的 checkpoints。

```shell
python tools/model_converters/upgrade_model_version.py ${IN_FILE} ${OUT_FILE} [-h] [--num-classes NUM_CLASSES]
```

### RegNet 模型转换至 MMDetection 模型

`tools/model_converters/regnet2mmdet.py` 将 pycls 编码的预训练 RegNet 模型转换为 MMDetection 风格。

```shell
python tools/model_converters/regnet2mmdet.py ${SRC} ${DST} [-h]
```

### Detectron ResNet 模型转换至 Pytorch 模型

`tools/model_converters/detectron2pytorch.py` 将 detectron 的原始预训练 RegNet 模型转换为 MMDetection 风格。

```shell
python tools/model_converters/detectron2pytorch.py ${SRC} ${DST} ${DEPTH} [-h]
```

### 制作发布用模型

`tools/model_converters/publish_model.py` 可用来制作一个发布用的模型。

在发布模型至 AWS 之前，你可能需要：

1. 将模型转换至 CPU 张量
2. 删除优化器状态
3. 计算 checkpoint 文件的 hash 值，并将 hash 号码记录至文件名。

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

样例：

```shell
python tools/model_converters/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```

最后输出的文件名如下所示： `faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

## 数据集转换

`tools/data_converters/` 提供了将 Cityscapes 数据集与 Pascal VOC 数据集转换至 COCO 数据集格式的工具

```shell
python tools/dataset_converters/cityscapes.py ${CITYSCAPES_PATH} [-h] [--img-dir ${IMG_DIR}] [--gt-dir ${GT_DIR}] [-o ${OUT_DIR}] [--nproc ${NPROC}]
python tools/dataset_converters/pascal_voc.py ${DEVKIT_PATH} [-h] [-o ${OUT_DIR}]
```

## 数据集下载

`tools/misc/download_dataset.py` 可以下载各类形如 COCO， VOC， LVIS 数据集。

```shell
python tools/misc/download_dataset.py --dataset-name coco2017
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name lvis
```

对于中国境内的用户，我们也推荐使用开源数据平台 [OpenDataLab](https://opendatalab.com/?source=OpenMMLab%20GitHub) 来获取这些数据集，以获得更好的下载体验:

- [COCO2017](https://opendatalab.com/COCO_2017/download?source=OpenMMLab%20GitHub)
- [VOC2007](https://opendatalab.com/PASCAL_VOC2007/download?source=OpenMMLab%20GitHub)
- [VOC2012](https://opendatalab.com/PASCAL_VOC2012/download?source=OpenMMLab%20GitHub)
- [LVIS](https://opendatalab.com/LVIS/download?source=OpenMMLab%20GitHub)

## 基准测试

### 鲁棒性测试基准

`tools/analysis_tools/test_robustness.py` 及 `tools/analysis_tools/robustness_eval.py` 帮助使用者衡量模型的鲁棒性。其核心思想来源于 [Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming](https://arxiv.org/abs/1907.07484)。如果你想了解如何在污损图像上评估模型的效果，以及参考该基准的一组标准模型，请参照 [robustness_benchmarking.md](robustness_benchmarking.md)。

### FPS 测试基准

`tools/analysis_tools/benchmark.py` 可帮助使用者计算 FPS，FPS 计算包括了模型向前传播与后处理过程。为了得到更精确的计算值，现在的分布式计算模式只支持一个 GPU。

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} tools/analysis_tools/benchmark.py \
    ${CONFIG} \
    [--checkpoint ${CHECKPOINT}] \
    [--repeat-num ${REPEAT_NUM}] \
    [--max-iter ${MAX_ITER}] \
    [--log-interval ${LOG_INTERVAL}] \
    --launcher pytorch
```

样例：假设你已经下载了 `Faster R-CNN` 模型 checkpoint 并放置在 `checkpoints/` 目录下。

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --launcher pytorch
```

## 更多工具

### 以某个评估标准进行评估

`tools/analysis_tools/eval_metric.py` 根据配置文件中的评估方式对 pkl 结果文件进行评估。

```shell
python tools/analysis_tools/eval_metric.py ${CONFIG} ${PKL_RESULTS} [-h] [--format-only] [--eval ${EVAL[EVAL ...]}]
                      [--cfg-options ${CFG_OPTIONS [CFG_OPTIONS ...]}]
                      [--eval-options ${EVAL_OPTIONS [EVAL_OPTIONS ...]}]
```

### 打印全部 config

`tools/misc/print_config.py` 可将所有配置继承关系展开，完全打印相应的配置文件。

```shell
python tools/misc/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

## 超参数优化

### YOLO Anchor 优化

`tools/analysis_tools/optimize_anchors.py` 提供了两种方法优化 YOLO 的 anchors。

其中一种方法使用 K 均值 anchor 聚类（k-means anchor cluster），源自 [darknet](https://github.com/AlexeyAB/darknet/blob/master/src/detector.c#L1421)。

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} --algorithm k-means --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} --output-dir ${OUTPUT_DIR}
```

另一种方法使用差分进化算法优化 anchors。

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} --algorithm differential_evolution --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} --output-dir ${OUTPUT_DIR}
```

样例：

```shell
python tools/analysis_tools/optimize_anchors.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py --algorithm differential_evolution --input-shape 608 608 --device cuda --output-dir work_dirs
```

你可能会看到如下结果：

```
loading annotations into memory...
Done (t=9.70s)
creating index...
index created!
2021-07-19 19:37:20,951 - mmdet - INFO - Collecting bboxes from annotation...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 117266/117266, 15874.5 task/s, elapsed: 7s, ETA:     0s

2021-07-19 19:37:28,753 - mmdet - INFO - Collected 849902 bboxes.
differential_evolution step 1: f(x)= 0.506055
differential_evolution step 2: f(x)= 0.506055
......

differential_evolution step 489: f(x)= 0.386625
2021-07-19 19:46:40,775 - mmdet - INFO Anchor evolution finish. Average IOU: 0.6133754253387451
2021-07-19 19:46:40,776 - mmdet - INFO Anchor differential evolution result:[[10, 12], [15, 30], [32, 22], [29, 59], [61, 46], [57, 116], [112, 89], [154, 198], [349, 336]]
2021-07-19 19:46:40,798 - mmdet - INFO Result saved in work_dirs/anchor_optimize_result.json
```

## 混淆矩阵

混淆矩阵是对检测结果的概览。
`tools/analysis_tools/confusion_matrix.py` 可对预测结果进行分析，绘制成混淆矩阵表。
首先，运行 `tools/test.py` 保存 `.pkl` 预测结果。
之后再运行：

```
python tools/analysis_tools/confusion_matrix.py ${CONFIG}  ${DETECTION_RESULTS}  ${SAVE_DIR} --show
```

最后你可以得到如图的混淆矩阵：

![confusion_matrix_example](https://user-images.githubusercontent.com/12907710/140513068-994cdbf4-3a4a-48f0-8fd8-2830d93fd963.png)

## COCO 分离和遮挡实例分割性能评估

对于最先进的目标检测器来说，检测被遮挡的物体仍然是一个挑战。
我们实现了论文 [A Tri-Layer Plugin to Improve Occluded Detection](https://arxiv.org/abs/2210.10046) 中提出的指标来计算分离和遮挡目标的召回率。

使用此评价指标有两种方法：

### 离线评测

我们提供了一个脚本对存储后的检测结果文件计算指标。

首先，使用 `tools/test.py` 脚本存储检测结果：

```shell
python tools/test.py ${CONFIG} ${MODEL_PATH} --out results.pkl
```

然后，运行 `tools/analysis_tools/coco_occluded_separated_recall.py` 脚本来计算分离和遮挡目标的掩码的召回率:

```shell
python tools/analysis_tools/coco_occluded_separated_recall.py results.pkl --out occluded_separated_recall.json
```

输出如下：

```
loading annotations into memory...
Done (t=0.51s)
creating index...
index created!
processing detection results...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 5000/5000, 109.3 task/s, elapsed: 46s, ETA:     0s
computing occluded mask recall...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 5550/5550, 780.5 task/s, elapsed: 7s, ETA:     0s
COCO occluded mask recall: 58.79%
COCO occluded mask success num: 3263
computing separated mask recall...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3522/3522, 778.3 task/s, elapsed: 5s, ETA:     0s
COCO separated mask recall: 31.94%
COCO separated mask success num: 1125

+-----------+--------+-------------+
| mask type | recall | num correct |
+-----------+--------+-------------+
| occluded  | 58.79% | 3263        |
| separated | 31.94% | 1125        |
+-----------+--------+-------------+
Evaluation results have been saved to occluded_separated_recall.json.
```

### 在线评测

我们实现继承自 `CocoMetic` 的 `CocoOccludedSeparatedMetric`。
要在训练期间评估分离和遮挡掩码的召回率，只需在配置中将 evaluator 类型替换为 `CocoOccludedSeparatedMetric`：

```python
val_evaluator = dict(
    type='CocoOccludedSeparatedMetric',  # 修改此处
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = val_evaluator
```

如果您使用了此指标，请引用论文：

```latex
@article{zhan2022triocc,
    title={A Tri-Layer Plugin to Improve Occluded Detection},
    author={Zhan, Guanqi and Xie, Weidi and Zisserman, Andrew},
    journal={British Machine Vision Conference},
    year={2022}
}
```

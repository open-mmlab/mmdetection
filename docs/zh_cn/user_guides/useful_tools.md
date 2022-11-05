## 日志分析

`tools/analysis_tools/analyze_logs.py` 利用指定的训练log文件绘制 loss/mAP 曲线图，
当第一次运行前请先运行 `pip install seaborn` 安装必要依赖.

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

使用 `tools/analysis_tools/analyze_results.py` 可计算每个图像 mAP，随后根据真实标注框与预测框的比较结果，展示或保存最高与最低 top-k 得分的预测图像

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

- `config` : model config 文件的地址。
- `prediction_path`:  使用`tools/test.py`输出的pickle格式结果文件。
- `show_dir`: 绘制标注框与预测框的图像保存地址。
- `--show`：决定是否展示绘制box后的图片，默认值为`False`。
- `--wait-time`: show时间的间隔，若为0表示持续显示。
- `--topk`: 根据最高或最低`topk`概率排序保存的图片数量，若不指定，默认设置为`20`。
- `--show-score-thr`: 能够展示的概率阈值，默认为`0`。
- `--cfg-options`: 如果指定，可根据指定键值对覆盖更新配置文件的对应选项

**样例**:
假设你已经通过 `tools/test.py` 得到了 pickle 格式的结果文件，其路径为 './result.pkl'。

1. 测试 Faster R-CNN 并可视化结果，保存图片至 `results/`

```shell
python tools/analysis_tools/analyze_results.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       result.pkl \
       results \
       --show
```

2. 测试 Faster R-CNN 并指定 top-k 参数为50，保存结果图片至 `results/`

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

`tools/analysis_tools/browse_dataset.py` 可帮助使用者浏览监测数据集（包括图像和标注），或保存图像至指定目录。

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

### 可视化模型

在可视化之前，需要先转换模型至 ONNX 格式，
[可参考此处](#convert-mmdetection-model-to-onnx-experimental)。
注意，现在只支持 RetinaNet，之后的版本将会支持其他模型
转换后的模型可以被其他工具可视化[Netron](https://github.com/lutzroeder/netron)。

### 可视化预测结果

如果你想要一个轻量 GUI 可视化检测结果，你可以参考 [DetVisGUI project](https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection)。
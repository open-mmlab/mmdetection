**我们在 `tools/` 目录下提供了很多有用的工具。**

## MOT 测试时参数搜索

`tools/analysis_tools/mot/mot_param_search.py` 可以搜索 MOT 模型中 `tracker` 的参数。
它与 `tools/test.py` 的使用方式相同，但配置上**有所不同**。

下面是修改配置的示例：

1. 定义要记录的期望评估指标。

   例如，你可以将 `evaluator` 定义为：

   ```python
   test_evaluator=dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
   ```

   当然，你也可以自定义 `test_evaluator` 中 `metric` 的内容。你可以自由选择 `['HOTA', 'CLEAR', 'Identity']` 中的一个或多个指标。

2. 定义要搜索的参数及其取值。

   假设你有一个 `tracker` 的配置如下：

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   如果你想要搜索 `tracker` 的参数，只需将其值改为一个列表，如下所示：

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   然后，脚本将测试一共12种情况并且记录结果。

## MOT 误差可视化

`tools/analysis_tools/mot/mot_error_visualize.py` 可以为多目标跟踪可视化错误。

该脚本需要推断的结果作为输入。默认情况下，**红色**边界框表示误检（false positive），**黄色**边界框表示漏检（false negative），**蓝色**边界框表示ID切换（ID switch）。

```
python tools/analysis_tools/mot/mot_error_visualize.py \
    ${CONFIG_FILE}\
    --input ${INPUT} \
    --result-dir ${RESULT_DIR} \
    [--output-dir ${OUTPUT}] \
    [--fps ${FPS}] \
    [--show] \
    [--backend ${BACKEND}]
```

`RESULT_DIR` 中包含了所有视频的推断结果，推断结果是一个 `txt` 文件。

可选参数：

- `OUTPUT`：可视化演示的输出。如果未指定，`--show` 是必选的，用于即时显示视频。
- `FPS`：输出视频的帧率。
- `--show`：是否即时显示视频。
- `BACKEND`：用于可视化边界框的后端。选项包括 `cv2` 和 `plt`。

## 浏览数据集

`tools/analysis_tools/mot/browse_dataset.py` 可以可视化训练数据集，以检查数据集配置是否正确。

**示例：**

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [--show-interval ${SHOW_INTERVAL}]
```

可选参数：

- `SHOW_INTERVAL`: 显示的间隔时间（秒）。
- `--show`: 是否即时显示图像。

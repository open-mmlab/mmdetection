# 了解可视化

## 本地的可视化

这一节将会展示如何使用本地的工具可视化detection/tracking的运行结果。

如果你想要画出预测结果的图像，你可以如下示例，将`TrackVisualizationHook` 中的特征设定为`draw=True` 。

```shell
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=True))
```

特别地，`TrackVisualizationHook`有如下参数：

- `draw`:是否画出预测结果，如果该参数被设置为False，就不会画出图像。该参数默认设置为False。

- `interval`: 可视化图的间隔。默认值为30。

- `score_thr`: 可视化bbox和masks的起点。默认值是0.3。

- `show`:是否展示绘制的图像。

- `wait_time`: 展示的时间间隔。默认为0。

- `test_out_dir`:在测试过程中保存绘制图像的目录。

- `backend_args`:实例化文件客户机的参数。默认值为`None `。

  在`TrackVisualizationHook`中，将调用`TrackLocalVisualizer `来实现MOT和VIS任务的可视化。我们将在下面介绍细节。

  你可以访问MMEngine获取 [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) 和 [Hook](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md)的更多细节。

  ### Tracking的可视化

  我们用`TrackLocalVisualizer`类实现跟踪可视化。你可以这样调用它。

  ```python
  visualizer = dict(type='TrackLocalVisualizer')
  ```

  visualizer有如下的参数：

  - `name`: 选择实例的名称。默认值为'visualizer'。Name of the instance. Defaults to 'visualizer'.
  - `image`: 需要处理的原图。默认为None。
  - `vis_backends`:可视化后端配置列表。默认为None。
  - `save_dir`:为所有存储后端保存文件目录。如果文件夹为空，后端储存仓将不会储存任何数据。
  - `line_width`:线的线宽。默认值为3。
  - `alpha`: bboxes或者mask的透明度。默认为0.8。

  这里是一个DeepSORT的可视化示例：

  ![test_img_89](https://user-images.githubusercontent.com/99722489/186062929-6d0e4663-0d8e-4045-9ec8-67e0e41da876.png)

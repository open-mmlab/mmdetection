# 了解可视化

## 本地的可视化

这一节将会展示如何使用本地的工具可视化 detection/tracking 的运行结果。

如果你想要画出预测结果的图像，你可以如下示例，将 `TrackVisualizationHook` 中的 draw 的参数设置为 `draw=True`。

```shell
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=True))
```

`TrackVisualizationHook` 共有如下参数：

- `draw`： 是否绘制预测结果。如果选择 False，将不会显示图像。该参数默认设置为 False。
- `interval`： 可视化的间隔。默认值为 30。
- `score_thr`： 确定是否可视化边界框和掩码的阈值。默认值是 0.3。
- `show`： 是否展示绘制的图像。默认不显示。
- `wait_time`： 展示的时间间隔(秒)。默认为 0。
- `test_out_dir`： 测试过程中绘制图像保存的目录。
- `backend_args`： 用于实例化文件客户端的参数。默认值为 `None `。

在 `TrackVisualizationHook` 中，将调用 `TrackLocalVisualizer` 来实现 MOT 和 VIS 任务的可视化。具体细节如下。

你可以通过 MMEngine 获取  [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md) 和  [Hook](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md) 的更多细节。

### Tracking 的可视化

我们使用 `TrackLocalVisualizer` 这个类以实现跟踪任务可视化。调用方式如下：

```python
visualizer = dict(type='TrackLocalVisualizer')
```

visualizer 共有如下的参数：

- `name`： 所选实例的名称。默认值为 ‘visualizer’。

- `image`： 用于绘制的原始图像。格式需要为 RGB。默认为 None。

- `vis_backends`： 可视化后端配置列表。默认为 None。

- `save_dir`： 所有后端存储的保存文件目录。如果为 None，后端将不会保存任何数据。

- `line_width`： 边框宽度。默认值为 3。

- `alpha`： 边界框和掩码的透明度。默认为 0.8。

这里提供了一个 DeepSORT 的可视化示例：

![test_img_89](https://user-images.githubusercontent.com/99722489/186062929-6d0e4663-0d8e-4045-9ec8-67e0e41da876.png)

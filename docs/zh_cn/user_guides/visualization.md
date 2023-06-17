# 可视化

在阅读本教程之前，建议先阅读 MMEngine 的 [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) 文档，以对 `Visualizer` 的定义和用法有一个初步的了解。

简而言之，`Visualizer` 在 MMEngine 中实现以满足日常可视化需求，并包含以下三个主要功能：

- 实现通用的绘图 API，例如 [`draw_bboxes`](mmengine.visualization.Visualizer.draw_bboxes) 实现了绘制边界框的功能，[`draw_lines`](mmengine.visualization.Visualizer.draw_lines) 实现了绘制线条的功能。
- 支持将可视化结果、学习率曲线、损失函数曲线以及验证精度曲线写入到各种后端中，包括本地磁盘以及常见的深度学习训练日志工具，例如 [TensorBoard](https://www.tensorflow.org/tensorboard) 和 [Wandb](https://wandb.ai/site)。
- 支持在代码的任何位置调用以可视化或记录模型在训练或测试期间的中间状态，例如特征图和验证结果。

基于 MMEngine 的 `Visualizer`，MMDet 提供了各种预构建的可视化工具，用户可以通过简单地修改以下配置文件来使用它们。

- `tools/analysis_tools/browse_dataset.py` 脚本提供了一个数据集可视化功能，可以在数据经过数据转换后绘制图像和相应的注释，具体描述请参见[`browse_dataset.py`](useful_tools.md#Visualization)。

- MMEngine实现了`LoggerHook`，使用`Visualizer`将学习率、损失和评估结果写入由`Visualizer`设置的后端。因此，通过修改配置文件中的`Visualizer`后端，例如修改为`TensorBoardVISBackend`或`WandbVISBackend`，可以实现日志记录到常用的训练日志工具，如`TensorBoard`或`WandB`，从而方便用户使用这些可视化工具来分析和监控训练过程。

- 在MMDet中实现了`VisualizerHook`，它使用`Visualizer`将验证或预测阶段的预测结果可视化或存储到由`Visualizer`设置的后端。因此，通过修改配置文件中的`Visualizer`后端，例如修改为`TensorBoardVISBackend`或`WandbVISBackend`，可以将预测图像存储到`TensorBoard`或`Wandb`中。

## 配置

由于使用了注册机制，在MMDet中我们可以通过修改配置文件来设置`Visualizer`的行为。通常，我们会在`configs/_base_/default_runtime.py`中为可视化器定义默认配置，详细信息请参见[配置教程](config.md)。

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```

基于上面的例子，我们可以看到`Visualizer`的配置由两个主要部分组成，即`Visualizer`类型和其使用的可视化后端`vis_backends`。

- 用户可直接使用`DetLocalVisualizer`来可视化支持任务的标签或预测结果。
- MMDet默认将可视化后端`vis_backend`设置为本地可视化后端`LocalVisBackend`，将所有可视化结果和其他训练信息保存在本地文件夹中。

## 存储

MMDet默认使用本地可视化后端[`LocalVisBackend`](mmengine.visualization.LocalVisBackend)，`VisualizerHook`和`LoggerHook`中存储的模型损失、学习率、模型评估精度和可视化信息，包括损失、学习率、评估精度将默认保存到`{work_dir}/{config_name}/{time}/{vis_data}`文件夹中。此外，MMDet还支持其他常见的可视化后端，例如`TensorboardVisBackend`和`WandbVisBackend`，您只需要在配置文件中更改`vis_backends`类型为相应的可视化后端即可。例如，只需在配置文件中插入以下代码块即可将数据存储到`TensorBoard`和`Wandb`中。

```Python
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),]
```

## 绘图

### 绘制预测结果

MMDet主要使用[`DetVisualizationHook`](mmdet.engine.hooks.DetVisualizationHook)来绘制验证和测试的预测结果，默认情况下`DetVisualizationHook`是关闭的，其默认配置如下。

```Python
visualization=dict( #用户可视化验证和测试结果
    type='DetVisualizationHook',
    draw=False,
    interval=1,
    show=False)
```

以下表格展示了`DetVisualizationHook`支持的参数。

|   参数   |                                       描述                                       |
| :------: | :------------------------------------------------------------------------------: |
|   draw   |          DetVisualizationHook通过enable参数打开和关闭，默认状态为关闭。          |
| interval | 控制在DetVisualizationHook启用时存储或显示验证或测试结果的间隔，单位为迭代次数。 |
|   show   |                         控制是否可视化验证或测试的结果。                         |

如果您想在训练或测试期间启用 `DetVisualizationHook` 相关功能和配置，您只需要修改配置文件，以 `configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py` 为例，同时绘制注释和预测，并显示图像，配置文件可以修改如下：

```Python
visualization = _base_.default_hooks.visualization
visualization.update(dict(draw=True, show=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/224883427-1294a7ba-14ab-4d93-9152-55a7b270b1f1.png" height="300"/>
</div>

`test.py`程序提供了`--show`和`--show-dir`参数，可以在测试过程中可视化注释和预测结果，而不需要修改配置文件，从而进一步简化了测试过程。

```Shell
# 展示测试结果
python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --show

# 指定存储预测结果的位置
python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --show-dir imgs/
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/224883427-1294a7ba-14ab-4d93-9152-55a7b270b1f1.png" height="300"/>
</div>

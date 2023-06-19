# Visualization

Before reading this tutorial, it is recommended to read MMEngine's [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) documentation to get a first glimpse of the `Visualizer` definition and usage.

In brief, the [`Visualizer`](mmengine.visualization.Visualizer) is implemented in MMEngine to meet the daily visualization needs, and contains three main functions:

- Implement common drawing APIs, such as [`draw_bboxes`](mmengine.visualization.Visualizer.draw_bboxes) which implements bounding box drawing functions, [`draw_lines`](mmengine.visualization.Visualizer.draw_lines) implements the line drawing function.
- Support writing visualization results, learning rate curves, loss function curves, and verification accuracy curves to various backends, including local disks and common deep learning training logging tools such as [TensorBoard](https://www.tensorflow.org/tensorboard) and [Wandb](https://wandb.ai/site).
- Support calling anywhere in the code to visualize or record intermediate states of the model during training or testing, such as feature maps and validation results.

Based on MMEngine's Visualizer, MMDet comes with a variety of pre-built visualization tools that can be used by the user by simply modifying the following configuration files.

- The `tools/analysis_tools/browse_dataset.py` script provides a dataset visualization function that draws images and corresponding annotations after Data Transforms, as described in [`browse_dataset.py`](useful_tools.md#Visualization).
- MMEngine implements `LoggerHook`, which uses `Visualizer` to write the learning rate, loss and evaluation results to the backend set by `Visualizer`. Therefore, by modifying the `Visualizer` backend in the configuration file, for example to ` TensorBoardVISBackend` or `WandbVISBackend`, you can implement logging to common training logging tools such as `TensorBoard` or `WandB`, thus making it easy for users to use these visualization tools to analyze and monitor the training process.
- The `VisualizerHook` is implemented in MMDet, which uses the `Visualizer` to visualize or store the prediction results of the validation or prediction phase into the backend set by the `Visualizer`, so by modifying the `Visualizer` backend in the configuration file, for example, to ` TensorBoardVISBackend` or `WandbVISBackend`, you can implement storing the predicted images to `TensorBoard` or `Wandb`.

## Configuration

Thanks to the use of the registration mechanism, in MMDet we can set the behavior of the `Visualizer` by modifying the configuration file. Usually, we define the default configuration for the visualizer in `configs/_base_/default_runtime.py`, see [configuration tutorial](config.md) for details.

```Python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
```

Based on the above example, we can see that the configuration of `Visualizer` consists of two main parts, namely, the type of `Visualizer` and the visualization backend `vis_backends` it uses.

- Users can directly use `DetLocalVisualizer` to visualize labels or predictions for support tasks.
- MMDet sets the visualization backend `vis_backend` to the local visualization backend `LocalVisBackend` by default, saving all visualization results and other training information in a local folder.

## Storage

MMDet uses the local visualization backend [`LocalVisBackend`](mmengine.visualization.LocalVisBackend) by default, and the model loss, learning rate, model evaluation accuracy and visualization The information stored in `VisualizerHook` and `LoggerHook`, including loss, learning rate, evaluation accuracy will be saved to the `{work_dir}/{config_name}/{time}/{vis_data}` folder by default. In addition, MMDet also supports other common visualization backends, such as `TensorboardVisBackend` and `WandbVisBackend`, and you only need to change the `vis_backends` type in the configuration file to the corresponding visualization backend. For example, you can store data to `TensorBoard` and `Wandb` by simply inserting the following code block into the configuration file.

```Python
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),]
```

## Plot

### Plot the prediction results

MMDet mainly uses [`DetVisualizationHook`](mmdet.engine.hooks.DetVisualizationHook) to plot the prediction results of validation and test, by default `DetVisualizationHook` is off, and the default configuration is as follows.

```Python
visualization=dict( # user visualization of validation and test results
    type='DetVisualizationHook',
    draw=False,
    interval=1,
    show=False)
```

The following table shows the parameters supported by `DetVisualizationHook`.

| Parameters |                                                  Description                                                  |
| :--------: | :-----------------------------------------------------------------------------------------------------------: |
|    draw    |      The DetVisualizationHook is turned on and off by the enable parameter, which is the default state.       |
|  interval  | Controls how much iteration to store or display the results of a val or test if VisualizationHook is enabled. |
|    show    |                           Controls whether to visualize the results of val or test.                           |

If you want to enable `DetVisualizationHook` related functions and configurations during training or testing, you only need to modify the configuration, take `configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py` as an example, draw annotations and predictions at the same time, and display the images, the configuration can be modified as follows

```Python
visualization = _base_.default_hooks.visualization
visualization.update(dict(draw=True, show=True))
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/224883427-1294a7ba-14ab-4d93-9152-55a7b270b1f1.png" height="300"/>
</div>

The `test.py` procedure is further simplified by providing the  `--show` and `--show-dir` parameters to visualize the annotation and prediction results during the test without modifying the configuration.

```Shell
# Show test results
python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --show

# Specify where to store the prediction results
python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --show-dir imgs/
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/224883427-1294a7ba-14ab-4d93-9152-55a7b270b1f1.png" height="300"/>
</div>

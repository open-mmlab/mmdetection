# Learn about Visualization

## Local Visualization

This section will present how to visualize the detection/tracking results with local visualizer.

If you want to draw prediction results, you can turn this feature on by setting `draw=True` in `TrackVisualizationHook` as follows.

```shell script
default_hooks = dict(visualization=dict(type='TrackVisualizationHook', draw=True))
```

Specifically, the `TrackVisualizationHook` has the following arguments:

- `draw`: whether to draw prediction results. If it is False, it means that no drawing will be done. Defaults to False.
- `interval`: The interval of visualization. Defaults to 30.
- `score_thr`: The threshold to visualize the bboxes and masks. Defaults to 0.3.
- `show`: Whether to display the drawn image. Default to False.
- `wait_time`: The interval of show (s). Defaults to 0.
- `test_out_dir`: directory where painted images will be saved in testing process.
- `backend_args`: Arguments to instantiate a file client. Defaults to `None`.

In the `TrackVisualizationHook`, `TrackLocalVisualizer` will be called to implement visualization for MOT and VIS tasks.
We will present the details below.
You can refer to MMEngine for more details about [Visualization](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) and [Hook](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/hook.md).

#### Tracking Visualization

We realize the tracking visualization with class `TrackLocalVisualizer`.
You can call it as follows.

```python
visualizer = dict(type='TrackLocalVisualizer')
```

It has the following arguments:

- `name`: Name of the instance. Defaults to 'visualizer'.
- `image`: The origin image to draw. The format should be RGB. Defaults to None.
- `vis_backends`: Visual backend config list. Defaults to None.
- `save_dir`: Save file dir for all storage backends. If it is None, the backend storage will not save any data.
- `line_width`: The linewidth of lines. Defaults to 3.
- `alpha`: The transparency of bboxes or mask. Defaults to 0.8.

Here is a visualization example of DeepSORT:

![test_img_89](https://user-images.githubusercontent.com/99722489/186062929-6d0e4663-0d8e-4045-9ec8-67e0e41da876.png)

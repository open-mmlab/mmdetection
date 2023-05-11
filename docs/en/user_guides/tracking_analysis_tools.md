**We provide lots of useful tools under the `tools/` directory.**

## MOT Test-time Parameter Search

`tools/analysis_tools/mot/mot_param_search.py` can search the parameters of the `tracker` in MOT models.
It is used as the same manner with `tools/test.py` but **different** in the configs.

Here is an example that shows how to modify the configs:

1. Define the desirable evaluation metrics to record.

   For example, you can define the `evaluator` as

   ```python
   test_evaluator=dict(type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
   ```

   Of course, you can also customize the content of `metric` in `test_evaluator`. You are free to choose one or more of `['HOTA', 'CLEAR', 'Identity']`.

2. Define the parameters and the values to search.

   Assume you have a tracker like

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=0.5,
           match_iou_thr=0.5
       )
   )
   ```

   If you want to search the parameters of the tracker, just change the value to a list as follow

   ```python
   model=dict(
       tracker=dict(
           type='BaseTracker',
           obj_score_thr=[0.4, 0.5, 0.6],
           match_iou_thr=[0.4, 0.5, 0.6, 0.7]
       )
   )
   ```

   Then the script will test the totally 12 cases and log the results.

## MOT Error Visualize

`tools/analysis_tools/mot/mot_error_visualize.py` can visualize errors for multiple object tracking.
This script needs the result of inference. By Default, the **red** bounding box denotes false positive, the **yellow** bounding box denotes the false negative and the **blue** bounding box denotes ID switch.

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

The `RESULT_DIR` contains the inference results of all videos and the inference result is a `txt` file.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `FPS`: FPS of the output video.
- `--show`: Whether show the video on the fly.
- `BACKEND`: The backend to visualize the boxes. Options are `cv2` and `plt`.

## Browse dataset

`tools/analysis_tools/mot/browse_dataset.py` can visualize the training dataset to check whether the dataset configuration is correct.

**Examples:**

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [--show-interval ${SHOW_INTERVAL}]
```

Optional arguments:

- `SHOW_INTERVAL`: The interval of show (s).
- `--show`: Whether show the images on the fly.

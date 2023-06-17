# Inference

We provide demo scripts to inference a given video or a folder that contains continuous images. The source codes are available [here](https://github.com/open-mmlab/mmdetection/tree/tracking/demo).

Note that if you use a folder as the input, the image names there must be  **sortable** , which means we can re-order the images according to the numbers contained in the filenames. We now only support reading the images whose filenames end with `.jpg`, `.jpeg` and `.png`.

## Inference MOT models

This script can inference an input video / images with a multiple object tracking or video instance segmentation model.

```shell
python demo/demo_mot.py \
    ${INPUTS}
    ${CONFIG_FILE} \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--detector ${DETECTOR_FILE}] \
    [--reid ${REID_FILE}] \
    [--score-thr ${SCORE_THR}] \
    [--device ${DEVICE}] \
    [--out ${OUTPUT}] \
    [--show]
```

The `INPUT` and `OUTPUT` support both _mp4 video_ format and the _folder_ format.

**Important:** For `DeepSORT`, `SORT`, `StrongSORT`, they need load the weight of the `reid` and the weight of the `detector` separately. Therefore, we use `--detector` and `--reid` to load weights. Other algorithms such as `ByteTrack`, `OCSORT` `QDTrack` `MaskTrackRCNN` and `Mask2Former` use `--checkpoint` to load weights.

Optional arguments:

- `CHECKPOINT_FILE`: The checkpoint is optional.
- `DETECTOR_FILE`: The detector is optional.
- `REID_FILE`: The reid is optional.
- `SCORE_THR`: The threshold of score to filter bboxes.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `--show`: Whether show the video on the fly.

**Examples of running mot model:**

```shell
# Example 1: do not specify --checkpoint to use --detector
python demo/demo_mot.py \
    demo/demo_mot.mp4 \
    configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --detector \
    https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --out mot.mp4

# Example 2: use --checkpoint
python demo/demo_mot.py \
    demo/demo_mot.mp4 \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth \
    --out mot.mp4
```

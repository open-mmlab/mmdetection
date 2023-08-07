# 预训练

我们提供了一些演示脚本去预训练一个给出的视频，或者是预训练包含一系列连续照片的文件夹。想要获取这个代码资源，请点击[这里](https://github.com/open-mmlab/mmdetection/tree/tracking/demo)。

如果输入为文件夹格式，你需要标明这点。图片命名应该**易于整理**，以便于你根据文件名字中包含的数字信息来重新调整图片的顺序。我们现在只支持`.jpg`，`.jpeg`和`.png`格式的图片。

## MOT models的预训练

这个脚本能够使用多任务跟踪或者视频实例分割方法来预训练一段输入的视频/一张图片。

```shell
python demo/mot_demo.py \
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

`输入内容`和`输出内容`支持_mp4 video_格式和文件格式。

**特别注意**：对于`DeepSORT`, `SORT`, `StrongSORT`,他们需要单独加载`reid`和`detector`的权重。因此，我们会使用 `--detector` 和`--reid` 来加载权重参数。例如`ByteTrack`, `OCSORT` `QDTrack` `MaskTrackRCNN` 以及`Mask2Former` 这样的其他算法则使用`--checkpoint` 来加载权重参数。

输入参数：

- `CHECKPOINT_FILE`: 可选择checkpoint。
- `DETECTOR_FILE`:  可选择detector。
- `REID_FILE`:  可选择reid。
- `SCORE_THR`:  bboxes的得分阈值。
- `DEVICE`: 预训练所需配置。可以选择 `cpu` ， `cuda:0`, 或者其他。
- `OUTPUT`: 输出结果可视化的示例。如果未指定，`--show` 将强制显示动态视频。
- `--show`: 是否即时显示视频。

**运行mot model的示例:**

```shell
#示例 1：不指定 -- 使用checkpoint --detector
python demo/mot_demo.py \
    demo/demo_mot.mp4 \
    configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --detector \
    https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth \
    --out mot.mp4

# 示例 2：使用 --checkpoint
python demo/mot_demo.py \
    demo/demo_mot.mp4 \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth \
    --out mot.mp4
```

# Test existing models on standard datasets

To evaluate a model's accuracy, one usually tests the model on some standard datasets, please refer to [dataset prepare guide](dataset_prepare.md) to prepare the dataset.

This section will show how to test existing models on supported datasets.

## Test existing models

We provide testing scripts for evaluating an existing model on the whole dataset (COCO, PASCAL VOC, Cityscapes, etc.).
The following testing environments are supported:

- single GPU
- CPU
- single node multiple GPUs
- multiple nodes

Choose the proper script to perform testing depending on the testing environment.

```shell
# Single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# Multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}]
```

`tools/dist_test.sh` also supports multi-node testing, but relies on PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment. Otherwise, you may encounter an error like `cannot connect to X server`.
- `--show-dir`: If specified, detection results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
- `--work-dir`: If specified, detection results containing evaluation metrics will be saved to the specified directory.
- `--cfg-options`:  If specified, the key-value pair optional cfg will be merged into config file

## Examples

Assuming that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test Faster R-CNN and visualize the results. Press any key for the next image.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/faster_rcnn).

   ```shell
   python tools/test.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --show
   ```

2. Test Faster R-CNN and save the painted images for future visualization.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/faster_rcnn).

   ```shell
   python tools/test.py \
       configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
       checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
       --show-dir faster_rcnn_r50_fpn_1x_results
   ```

3. Test Faster R-CNN on PASCAL VOC (without saving the test results).
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/pascal_voc).

   ```shell
   python tools/test.py \
       configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
       checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth
   ```

4. Test Mask R-CNN with 8 GPUs, and evaluate.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/mask_rcnn).

   ```shell
   ./tools/dist_test.sh \
       configs/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --out results.pkl
   ```

5. Test Mask R-CNN with 8 GPUs, and evaluate the metric **class-wise**.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/mask_rcnn).

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8 \
       --out results.pkl \
       --cfg-options test_evaluator.classwise=True
   ```

6. Test Mask R-CNN on COCO test-dev with 8 GPUs, and generate JSON files for submitting to the official evaluation server.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/mask_rcnn).

   Replace the original test_evaluator and test_dataloader with test_evaluator and test_dataloader in the comment in [config](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/_base_/datasets/coco_instance.py) and run:

   ```shell
   ./tools/dist_test.sh \
       configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
       checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
       8
   ```

   This command generates two JSON files `./work_dirs/coco_instance/test.bbox.json` and `./work_dirs/coco_instance/test.segm.json`.

7. Test Mask R-CNN on Cityscapes test with 8 GPUs, and generate txt and png files for submitting to the official evaluation server.
   Config and checkpoint files are available [here](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/cityscapes).

   Replace the original test_evaluator and test_dataloader with test_evaluator and test_dataloader in the comment in [config](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/_base_/datasets/cityscapes_instance.py) and run:

   ```shell
   ./tools/dist_test.sh \
       configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes.py \
       checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
       8
   ```

   The generated png and txt would be under `./work_dirs/cityscapes_metric/` directory.

## Test without Ground Truth Annotations

MMDetection supports to test models without ground-truth annotations using `CocoDataset`. If your dataset format is not in COCO format, please convert them to COCO format. For example, if your dataset format is VOC, you can directly convert it to COCO format by the [script in tools.](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/tools/dataset_converters/pascal_voc.py) If your dataset format is Cityscapes, you can directly convert it to COCO format by the [script in tools.](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/tools/dataset_converters/cityscapes.py) The rest of the formats can be converted using [this script](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/tools/dataset_converters/images2coco.py).

```shel
python tools/dataset_converters/images2coco.py \
    ${IMG_PATH} \
    ${CLASSES} \
    ${OUT} \
    [--exclude-extensions]
```

arguments：

- `IMG_PATH`: The root path of images.
- `CLASSES`: The text file with a list of categories.
- `OUT`: The output annotation json file name. The save dir is in the same directory as `IMG_PATH`.
- `exclude-extensions`: The suffix of images to be excluded, such as 'png' and 'bmp'.

After the conversion is complete, you need to replace the original test_evaluator and test_dataloader with test_evaluator and test_dataloader in the comment in [config](https://github.com/open-mmlab/mmdetection/tree/dev-3.x/configs/_base_/datasets/coco_detection.py)(find which dataset in 'configs/_base_/datasets' the current config corresponds to) and run:

```shell
# Single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--show]

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# Multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--show]
```

Assuming that the checkpoints in the [model zoo](https://mmdetection.readthedocs.io/en/latest/modelzoo_statistics.html) have been downloaded to the directory `checkpoints/`, we can test Mask R-CNN on COCO test-dev with 8 GPUs, and generate JSON files using the following command.

```sh
./tools/dist_test.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8
```

This command generates two JSON files `./work_dirs/coco_instance/test.bbox.json` and `./work_dirs/coco_instance/test.segm.json`.

## Batch Inference

MMDetection supports inference with a single image or batched images in test mode. By default, we use single-image inference and you can use batch inference by modifying `samples_per_gpu` in the config of test data. You can do that either by modifying the config as below.

```shell
data = dict(train_dataloader=dict(...), val_dataloader=dict(...), test_dataloader=dict(batch_size=2, ...))
```

Or you can set it through `--cfg-options` as `--cfg-options test_dataloader.batch_size=2`

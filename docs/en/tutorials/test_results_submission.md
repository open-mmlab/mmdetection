# Tutorial 11: Test Results Submission

## Panoptic segmentation test results submission

The following sections introduce how to produce the prediction results of panoptic segmentation models on the COCO test-dev set and submit the predictions to [COCO evaluation server](https://competitions.codalab.org/competitions/19507).

### Prerequisites

- Download [COCO test dataset images](http://images.cocodataset.org/zips/test2017.zip), [testing image info](http://images.cocodataset.org/annotations/image_info_test2017.zip), and [panoptic train/val annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip), then unzip them, put 'test2017' to `data/coco/`, put json files and annotation files to `data/coco/annotations/`.

- Run the following code to update category information in testing image info. Since the attribute `isthing` is missing in category information of 'image_info_test-dev2017.json', we need to update it with the category information in 'panoptic_val2017.json'.

```shell
python tools/misc/gen_coco_panoptic_test_info.py data/coco/annotations
```

After completing the above preparations, your directory structure of `data` should be like this:

```text
data
`-- coco
    |-- annotations
    |   |-- image_info_test-dev2017.json
    |   |-- image_info_test2017.json
    |   |-- panoptic_image_info_test-dev2017.json
    |   |-- panoptic_train2017
    |   |-- panoptic_train2017.json
    |   |-- panoptic_val2017
    |   `-- panoptic_val2017.json
    `-- test2017
```

### Inference on coco test-dev

Three ways to perform inference on `test2017`.

```shell
# test with single gpu
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --format-only \
    --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
    --eval-options jsonfile_prefix=${WORK_DIR}/results

# test with four gpus
CUDA_VISIBLE_DEVICES=0,1,3,4 bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    4 \ # four gpus
    --format-only \
    --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
    --eval-options jsonfile_prefix=${WORK_DIR}/results

# test with slurm
GPUS=8 tools/slurm_test.sh \
    ${Partition} \
    ${JOB_NAME} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --format-only \
    --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
    --eval-options jsonfile_prefix=${WORK_DIR}/results

```

Example

Suppose we perform inference on `test2017` with maskformer-r50.

```shell
# test with single gpu
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/maskformer/maskformer_r50_mstrain_16x1_75e_coco.py \
    checkpoints/maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth \
    --format-only \
    --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
    --eval-options jsonfile_prefix=work_dirs/maskformer/results

```

### Rename files and zip results

After inference, we will get panoptic segmentation results (a json file and a directory where the masks are stored) in `WORK_DIR`. We rename them according to naming convention described on [COCO's Website](https://cocodataset.org/#upload). Finally, we need to compress the json and the directory where the masks are stored into a zip file, and rename the zip file according to the naming convention. Note that the zip file should **directly** contains the above two files.

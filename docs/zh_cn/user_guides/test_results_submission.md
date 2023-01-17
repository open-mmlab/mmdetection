# 提交测试结果

## 全景分割测试结果提交

下面几节介绍如何在 COCO 测试开发集上生成泛视分割模型的预测结果，并将预测提交到 [COCO评估服务器](https://competitions.codalab.org/competitions/19507)

### 前提条件

- 下载 [COCO测试数据集图像](http://images.cocodataset.org/zips/test2017.zip)，[测试图像信息](http://images.cocodataset.org/annotations/image_info_test2017.zip)，和[全景训练/相关注释](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)，然后解压缩它们，把 `test2017` 放到 `data/coco/`，把 json 文件和注释文件放到 `data/coco/annotations/` 。

```shell
# 假设 data/coco/ 不存在
mkdir -pv data/coco/
# 下载 test2017
wget -P data/coco/ http://images.cocodataset.org/zips/test2017.zip
wget -P data/coco/ http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -P data/coco/ http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
# 解压缩它们
unzip data/coco/test2017.zip -d data/coco/
unzip data/coco/image_info_test2017.zip -d data/coco/
unzip data/coco/panoptic_annotations_trainval2017.zip -d data/coco/
# 删除 zip 文件(可选)
rm -rf data/coco/test2017.zip data/coco/image_info_test2017.zip data/coco/panoptic_annotations_trainval2017.zip
```

- 运行以下代码更新测试图像信息中的类别信息。由于 `image_info_test-dev2017.json` 的类别信息中缺少属性 `isthing` ，我们需要用 `panoptic_val2017.json` 中的类别信息更新它。

```shell
python tools/misc/gen_coco_panoptic_test_info.py data/coco/annotations
```

在完成上述准备之后，你的 `data` 目录结构应该是这样:

```text
data
`-- coco
    |-- annotations
    |   |-- image_info_test-dev2017.json
    |   |-- image_info_test2017.json
    |   |-- panoptic_image_info_test-dev2017.json
    |   |-- panoptic_train2017.json
    |   |-- panoptic_train2017.zip
    |   |-- panoptic_val2017.json
    |   `-- panoptic_val2017.zip
    `-- test2017
```

### coco 测试开发的推理

要在 coco test-dev 上进行推断，我们应该首先更新 `test_dataloder` 和 `test_evaluator` 的设置。有两种方法可以做到这一点:1. 在配置文件中更新它们;2. 在命令行中更新它们。

#### 在配置文件中更新它们

相关的设置在 `configs/_base_/datasets/ coco_panoptical .py` 的末尾，如下所示。

```python
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/panoptic_image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoPanopticMetric',
    format_only=True,
    ann_file=data_root + 'annotations/panoptic_image_info_test-dev2017.json',
    outfile_prefix='./work_dirs/coco_panoptic/test')
```

以下任何一种方法都可以用于更新 coco test-dev 集上的推理设置

情况1:直接取消注释 `configs/_base_/datasets/ coco_panoptical .py` 中的设置。

情况2:将以下设置复制到您现在使用的配置文件中。

```python
test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/panoptic_image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/', _delete_=True)))
test_evaluator = dict(
    format_only=True,
    ann_file=data_root + 'annotations/panoptic_image_info_test-dev2017.json',
    outfile_prefix='./work_dirs/coco_panoptic/test')
```

然后通过以下命令对 coco test-dev et 进行推断。

```shell
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE}
```

#### 在命令行中更新它们

coco test-dev 上更新相关设置和推理的命令如下所示。

```shell
# 用一个 gpu 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/panoptic_image_info_test-dev2017.json \
    test_dataloader.dataset.data_prefix.img=test2017 \
    test_dataloader.dataset.data_prefix._delete_=True \
    test_evaluator.format_only=True \
    test_evaluator.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json \
    test_evaluator.outfile_prefix=${WORK_DIR}/results
# 用四个 gpu 测试
CUDA_VISIBLE_DEVICES=0,1,3,4 bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    8 \  # eights gpus
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/panoptic_image_info_test-dev2017.json \
    test_dataloader.dataset.data_prefix.img=test2017 \
    test_dataloader.dataset.data_prefix._delete_=True \
    test_evaluator.format_only=True \
    test_evaluator.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json \
    test_evaluator.outfile_prefix=${WORK_DIR}/results
# 用 slurm 测试
GPUS=8 tools/slurm_test.sh \
    ${Partition} \
    ${JOB_NAME} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/panoptic_image_info_test-dev2017.json \
    test_dataloader.dataset.data_prefix.img=test2017 \
    test_dataloader.dataset.data_prefix._delete_=True \
    test_evaluator.format_only=True \
    test_evaluator.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json \
    test_evaluator.outfile_prefix=${WORK_DIR}/results
```

例子:假设我们使用预先训练的带有 ResNet-50 骨干网的 MaskFormer 对 `test2017` 执行推断。

```shell
# 单 gpu 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/maskformer/maskformer_r50_mstrain_16x1_75e_coco.py \
    checkpoints/maskformer_r50_mstrain_16x1_75e_coco_20220221_141956-bc2699cb.pth \
    --cfg-options \
    test_dataloader.dataset.ann_file=annotations/panoptic_image_info_test-dev2017.json \
    test_dataloader.dataset.data_prefix.img=test2017 \
    test_dataloader.dataset.data_prefix._delete_=True \
    test_evaluator.format_only=True \
    test_evaluator.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json \
    test_evaluator.outfile_prefix=work_dirs/maskformer/results
```

### 重命名文件并压缩结果

推理之后，全景分割结果(一个 json 文件和一个存储掩码的目录)将在 `WORK_DIR` 中。我们应该按照 [COCO's Website](https://cocodataset.org/#upload)上的命名约定重新命名它们。最后，我们需要将 json 和存储掩码的目录压缩到 zip 文件中，并根据命名约定重命名该 zip 文件。注意， zip 文件应该**直接**包含上述两个文件。

重命名文件和压缩结果的命令:

```shell
# 在 WORK_DIR 中，我们有 panoptic 分割结果: 'panoptic' 和 'results. panoptical .json'。
cd ${WORK_DIR}
# 将 '[algorithm_name]' 替换为您使用的算法名称
mv ./panoptic ./panoptic_test-dev2017_[algorithm_name]_results
mv ./results.panoptic.json ./panoptic_test-dev2017_[algorithm_name]_results.json
zip panoptic_test-dev2017_[algorithm_name]_results.zip -ur panoptic_test-dev2017_[algorithm_name]_results panoptic_test-dev2017_[algorithm_name]_results.json
```

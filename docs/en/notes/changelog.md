# Changelog of v3.x

## v3.0.0rc1 (26/9/2022)

### Highlights

- Release a high-precision, low-latency single-stage object detector [RTMDet](configs/rtmdet).

#### Bug Fixes

- Fix UT to be compatible with PyTorch 1.6 (#8707)
- Fix `NumClassCheckHook` bug when model is wrapped (#8794)
- Update the right URL of R-50-FPN with BoundedIoULoss (#8805)
- Fix potential bug of indices in RandAugment (#8826)
- Fix some types and links (#8839, #8820, #8793, #8868)
- Fix incorrect background fill values in `FSAF` and `RepPoints` Head (#8813)

#### Improvements

- Refactored anchor head and base head with `box type` (#8625)
- Refactored `SemiBaseDetector` and `SoftTeacher` (#8786)
- Add list to dict keys to avoid modify loss dict (#8828)
- Update `analyze_results.py` , `analyze_logs.py` and `loading.py` (#8430, #8402, #8784)
- Support dump results in `test.py` (#8814)
- Check empty predictions in `DetLocalVisualizer._draw_instances` (#8830)
- Fix `floordiv` warning in `SOLO` (#8738)

#### Contributors

A total of 16 developers contributed to this release.

Thanks @ZwwWayne, @jbwang1997, @Czm369, @ice-tong, @Zheng-LinXiao, @chhluo, @RangiLyu, @liuyanyi, @wanghonglie, @levan92, @JiayuXu0, @nye0, @hhaAndroid, @xin-li-67, @shuxp, @zytx121

## v3.0.0rc0 (31/8/2022)

We are excited to announce the release of MMDetection 3.0.0rc0. MMDet 3.0.0rc0 is the first version of MMDetection 3.x, a part of the OpenMMLab 2.0 projects. Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMDet 3.x unifies the interfaces of the dataset, models, evaluation, and visualization with faster training and testing speed. It also provides a general semi-supervised object detection framework and strong baselines.

### Highlights

1. **New engine**. MMDet 3.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a universal and powerful runner that allows more flexible customizations and significantly simplifies the entry points of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMDet 3.x unifies and refactors the interfaces and internal logic of training, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logic to allow the emergence of multi-task/modality algorithms.

3. **Faster speed**. We optimize the training and inference speed for common models and configurations, achieving a faster or similar speed than [Detection2](https://github.com/facebookresearch/detectron2/). Model details of benchmark will be updated in [this note](./benchmark.md#comparison-with-detectron2).

4. **General semi-supervised object detection**. Benefitting from the unified interfaces, we support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.

5. **Strong baselines**. We release strong baselines of many popular models to enable fair comparisons among state-of-the-art models.

6. **New features and algorithms**:

   - Enable all the single-stage detectors to serve as region proposal networks
   - [SoftTeacher](https://arxiv.org/abs/2106.09018)
   - [the updated CenterNet](https://arxiv.org/abs/2103.07461)

7. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection.readthedocs.io/en/3.x/).

### Breaking Changes

MMDet 3.x has undergone significant changes for better design, higher efficiency, more flexibility, and more unified interfaces.
Besides the changes in API, we briefly list the major breaking changes in this section.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.
Users can also refer to the [API doc](https://mmdetection.readthedocs.io/en/3.x/) for more details.

#### Dependencies

- MMDet 3.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMDet 3.x is not guaranteed.
- MMDet 3.x relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models of OpenMMLab and is the core dependency of OpenMMLab 2.0 projects. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMDet 3.x relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMDet 3.x relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provides pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated since 2.0.0rc0.

#### Training and testing

- MMDet 3.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of the dataset, model, evaluation, and visualizer. Therefore, MMDet 3.x no longer maintains the building logic of those modules in `mmdet.train.apis` and `tools/train.py`. Those codes have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic to that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum schedules have been migrated from Hook to [Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structure to ease the understanding of the components in the runner. Users can read the [config example of MMDet 3.x](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. The names of checkpoints are not updated for now as there is no BC-breaking of model weights between MMDet 3.x and 2.x. We will progressively replace all the model weights with those trained in MMDet 3.x. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Dataset

The Dataset classes implemented in MMDet 3.x all inherit from the `BaseDetDataset`, which inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). In addition to the changes in interfaces, there are several changes in Dataset in MMDet 3.x.

- All the datasets support serializing the internal data list to reduce the memory when multiple workers are built for data loading.
- The internal data structure in the dataset is changed to be self-contained (without losing information like class names in MMDet 2.x) while keeping simplicity.
- The evaluation functionality of each dataset has been removed from the dataset so that some specific evaluation metrics like COCO AP can be used to evaluate the prediction on other datasets.

#### Data Transforms

The data transforms in MMDet 3.x all inherits from `BaseTransform` in MMCV>=2.0.0rc0, which defines a new convention in OpenMMLab 2.0 projects.
Besides the interface changes, there are several changes listed below:

- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms to simplify and clarify the usages.
- The format of data dict processed by each data transform is changed according to the new data structure of dataset.
- Some inefficient data transforms (e.g., normalization and padding) are moved into data preprocessor of model to improve data loading and training speed.
- The same data transforms in different OpenMMLab 2.0 libraries have the same augmentation implementation and the logic given the same arguments, i.e., `Resize` in MMDet 3.x and MMSeg 1.x will resize the image in the exact same manner given the same arguments.

#### Model

The models in MMDet 3.x all inherit from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLab 2.0 projects.
Users can refer to [the tutorial of the model in MMengine](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) for more details.
Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMDet 3.x.
  Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a list of image tensors, and `data_samples` contains other information of the current data sample such as ground truths, region proposals, and model predictions. In this way, different tasks in MMDet 3.x can share the same input arguments, which makes the models more general and suitable for multi-task learning and some flexible training paradigms like semi-supervised learning.
- The model has a data preprocessor module, which is used to pre-process the input data of the model. In MMDet 3.x, the data preprocessor usually does the necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of the model has been changed. In MMdet 2.x, model uses `forward_train`, `forward_test`, `simple_test`, and `aug_test` to deal with different model forward logics. In MMDet 3.x and OpenMMLab 2.0, the forward function has three modes: 'loss', 'predict', and 'tensor' for training, inference, and tracing or other purposes, respectively.
  The forward function calls `self.loss`, `self.predict`, and `self._forward` given the modes 'loss', 'predict', and 'tensor', respectively.

#### Evaluation

The evaluation in MMDet 2.x strictly binds with the dataset. In contrast, MMDet 3.x decomposes the evaluation from dataset so that all the detection datasets can evaluate with COCO AP and other metrics implemented in MMDet 3.x.
MMDet 3.x mainly implements corresponding metrics for each dataset, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
Users can build an evaluator in MMDet 3.x to conduct offline evaluation, i.e., evaluate predictions that may not produce in MMDet 3.x with the dataset as long as the dataset and the prediction follow the dataset conventions. More details can be found in the [tutorial in mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html).

#### Visualization

The functions of visualization in MMDet 2.x are removed. Instead, in OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMDet 3.x implements `DetLocalVisualizer` to allow visualization of ground truths, model predictions, feature maps, etc., at any place. It also supports sending the visualization data to any external visualization backends such as Tensorboard.

### Improvements

- Optimized training and testing speed of FCOS, RetinaNet, Faster R-CNN, Mask R-CNN, and Cascade R-CNN. The training speed of those models with some common training strategies is also optimized, including those with synchronized batch normalization and mixed precision training.
- Support mixed precision training of all the models. However, some models may get undesirable performance due to some numerical issues. We will update the documentation and list the results (accuracy of failure) of mixed precision training.
- Release strong baselines of some popular object detectors. Their accuracy and pre-trained checkpoints will be released.

### Bug Fixes

- DeepFashion dataset: the config and results have been updated.

### New Features

1. Support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.
2. Enable all the single-stage detectors to serve as region proposal networks. We give [an example of using FCOS as RPN](../user_guides/single_stage_as_rpn.md).
3. Support a semi-supervised object detection algorithm: [SoftTeacher](https://arxiv.org/abs/2106.09018).
4. Support [the updated CenterNet](https://arxiv.org/abs/2103.07461).
5. Support data structures `HorizontalBoxes` and `BaseBoxes` to encapsulate different kinds of bounding boxes. We are migrating to use data structures of boxes to replace the use of pure tensor boxes. This will unify the usages of different kinds of bounding boxes in MMDet 3.x and MMRotate 1.x to simplify the implementation and reduce redundant codes.

### Planned changes

We list several planned changes of MMDet 3.0.0rc0 so that the community could more comprehensively know the progress of MMDet 3.x. Feel free to create a PR, issue, or discussion if you are interested, have any suggestions and feedback, or want to participate.

1. Test-time augmentation: which is supported in MMDet 2.x, is not implemented in this version due to the limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in Jupyter Notebook or Colab: more useful tools that are implemented in the `tools` directory will have their python interfaces so that they can be used in Jupyter Notebook, Colab, and downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMDet 3.x.
5. Wandb visualization: MMDet 2.x supports data visualization since v2.25.0, which has not been migrated to MMDet 3.x for now. Since WandB provides strong visualization and experiment management capabilities, a `DetWandBVisualizer` and maybe a hook are planned to fully migrate those functionalities from MMDet 2.x.
6. Full support of WiderFace dataset (#8508) and Fast R-CNN: we are verifying their functionalities and will fix related issues soon.
7. Migrate DETR-series algorithms (#8655, #8533) and YOLOv3 on IPU (#8552) from MMDet 2.x.

### Contributors

A total of 11 developers contributed to this release.
Thanks @shuxp, @wanghonglie, @Czm369, @BIGWangYuDong, @zytx121, @jbwang1997, @chhluo, @jshilong, @RangiLyu, @hhaAndroid, @ZwwWayne

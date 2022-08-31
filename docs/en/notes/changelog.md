# Changelog of v3.x

## v3.0.0rc0 (31/8/2022)

We are excited to announce the release of MMDetection 3.0.0rc0.
MMDet 3.0.0rc0 is the first version of MMDetection 3.x, a part of the OpenMMLab 2.x projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMDet 3.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.
It also provides a general semi-supervised object detection framework, and more strong baselines.

### Highlights

1. **New engines**. MMDet 3.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.x projects, MMDet 3.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.x projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Faster speed**. We optimize the training and inference speed for common models, achieving faster or similar speed in comparison with [Detection2](https://github.com/facebookresearch/detectron2/). Please refer to [benchmark](./benchmark.md#comparison-with-detectron2) for details.

4. **General semi-supervised object detection**. Benefitting from the unified interfaces, we support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.

5. **Strong baselines**. We release strong baselines of many popular models to enable fair comparisons among state-of-the-art models.

6. **New features and algorithms**:
    - Enable all the single-stage detectors to serve as region proposal networks
    - [SoftTeacher](https://arxiv.org/abs/2106.09018)
    - [the updated CenterNet](https://arxiv.org/abs/2103.07461)

7. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection.readthedocs.io/en/3.x/).

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Training and testing

- MMDet 3.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMDet 3.x is not guaranteed.
- MMDet 3.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMDet 3.x no longer maintains the building logics of those modules in `mmdet.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can read the [config example of mmdet](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.x projects. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Components

- Dataset
- Data Transforms
- Model
- Evaluation
- Visualization

### Improvements

- Improved training and testing speed of FCOS, RetinaNet, Faster R-CNN, Mask R-CNN, and Cascade R-CNN. The training speed of those models with some common training strategies are also improved, including those with synchronized batch normalization and mixed precision training.
- Support mixed precision training of all the models. However, some models may got Nan results due to some numerical issues. We will update the documentation and list their results (accuracy of failure) of mixed precision training.
- Release strong baselines of some popular object detectors. Their accuracy and pre-trained checkpoints will be released.

### Bug Fixes

- DeepFashion dataset: the config and results have been updated.

### New Features

1. Support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.
2. Enable all the single-stage detectors to serve as region proposal networks. We give [an example of using FCOS as RPN](../user_guides/single_stage_as_rpn.md).
3. Support a semi-supervised object detection algorithm: [SoftTeacher](https://arxiv.org/abs/2106.09018).
4. Support [the updated CenterNet](https://arxiv.org/abs/2103.07461).
5. Support data structures `HorizontalBoxes` and `BaseBoxes` to encapsulate different kinds of bounding boxes. We are migrating to use data structures of boxes to replace the use of pure tensor boxes. This will unify the usages of different kinds of bounding boxes in MMDet 3.x and MMRotate 1.x to simplify the implementation and reduce redundant codes.

### Ongoing changes

1. Test-time augmentation: which is supported in MMDet 2.x, is not implemented in this version due to limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMDet 3.x.
5. WiderFace dataset, and Fast R-CNN support: we are verifying their functionalities and will fix related issues soon.

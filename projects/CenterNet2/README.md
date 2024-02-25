# CenterNet2

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This is an implementation of [CenterNet2](https://github.com/xingyizhou/CenterNet2) based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/3.x), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

CenterNet2 is a two-stage object detectors that use class-agnostic one-stage detectors as the proposal network. First stage estimates object probabilities, second stage conditionally classifies objects. Resulting detector is faster and more accurate than both traditional two-stage detectors (fewer proposals required), and one-stage detectors (lighter first stage head).

<p align="center"> <img src='docs/centernet2_teaser.jpg' align="center" height="150px"> </p>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/CenterNet2/configs/centernet2-cascade_r50_fpn_1x_coco.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/CenterNet2/configs/centernet2-cascade_r50_fpn_1x_coco.py
```

### Testing commands

In MMDetection's root directory, run the following command to test the model:

```bash
python tools/test.py projects/CenterNet2/configs/centernet2-cascade_r50_fpn_1x_coco.py ${CHECKPOINT_PATH}
```

## Results

<!-- List the results as usually done in other model's README. [Example](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/README.md#results-and-models)
You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

Here we provide the baseline version of CenterNet2 with ResNet50 backbone.
To find more variants, please visit the [official model zoo](https://github.com/xingyizhou/CenterNet2#main-results).

|                           Method                            | Backbone | Pretrained Model |  Training set  |   Test set   | #epoch | box AP |         Download         |
| :---------------------------------------------------------: | :------: | :--------------: | :------------: | :----------: | :----: | :----: | :----------------------: |
| [CenterNet2](configs/centernet2-cascade_r50_fpn_1x_coco.py) |   R-50   |        -         | COCO2017 Train | COCO2017 Val |   12   | 0.8853 | [model](<>) \| [log](<>) |

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@article{Zhou_2021,
   title={Probabilistic two-stage detection},
   journal={arxiv},
   author={Xingyi Zhou, Vladlen Koltun, Philipp Krähenbühl},
   year={2021},
   month={Mar},
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [ ] A full README

    <!-- As this template does. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/mmdet/datasets/transforms/transforms.py#L41-L169) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/tests/test_datasets/test_transforms/test_transforms.py#L35-L88) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.

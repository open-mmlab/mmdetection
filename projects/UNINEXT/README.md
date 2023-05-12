# UNINEXT

> [**Universal Instance Perception as Object Discovery and Retrieval**](https://arxiv.org/abs/2303.06674),
> Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, Huchuan Lu
> *CVPR 2023*

## Abstract

This is an implementation of [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT) based on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).
<br>
Object-centric understanding is one of the most essential and challenging problems in computer vision. In this work, we mainly discuss 10 sub-tasks, distributed on the vertices of the cube shown in the above figure. Since all these tasks aim to perceive instances of certain properties, UNINEXT reorganizes them into three types according to the different input prompts:

- Category Names
  - Object Detection
  - Instance Segmentation
  - Multiple Object Tracking (MOT)
  - Multi-Object Tracking and Segmentation (MOTS)
  - Video Instance Segmentation (VIS)
- Language Expressions
  - Referring Expression Comprehension (REC)
  - Referring Expression Segmentation (RES)
  - Referring Video Object Segmentation (R-VOS)
- Target Annotations
  - Single Object Tracking (SOT)
  - Video Object Segmentation (VOS)

Then UNINEXT propose a unified prompt-guided object discovery and retrieval formulation
to solve all the above tasks and achieves superior performance on 20 challenging benchmarks.
<br>

</div>

**Note**

1. Due to time constraints, this project only realizes VIS tasks testing
2. the training and testing of other tasks will be released in the future

## Usage

### Environment installation

```bash
pip3 install --user shapely==1.7.1
pip3 install --user git+https://github.com/XD7479/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
pip3 install --user git+https://github.com/lvis-dataset/lvis-api.git
pip3 install --user jpeg4py visdom easydict scikit-image
pip3 install --user transformers tikzplotlib motmetrics
```

### Model conversion

Language Model (BERT-base)

```bash
mkdir -p projects/UNINEXT/uninext/bert-base-uncased
cd projects/UNINEXT/uninext/bert-base-uncased
wget -c https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget -c https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -c https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
cd ../../../..
```

Vision Model
Firstly, download UNINEXT R50 [weights](https://maildluteducn-my.sharepoint.com/:f:/g/personal/yan_bin_mail_dlut_edu_cn/ErbTZCzv0vJAoIMwa90_3qoBOFbHIJJTVxI58-kk2nfkhw?e=4qvjrR)

Then, convert weights, please use the following command

```bash
python projects/UNINEXT/convert_weights.py --original-weight ${CHECKPOINT_PATH} --out-weight ${CHECKPOINT_PATH}
```

### Testing commands

In MMDetection's root directory, run the following command to test the model:

```bash
bash tools/dist_test_tracking.sh projects/UNINEXT/configs/uninext_r50-8e_youtubevis2019.py 8 --checkpoint ${CHECKPOINT_PATH}
```

## Results

Based on mmdetection, this project aligns the VIS task test accuracy of the  [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT).
<br>
If you want to reproduce the test results, you need to convert model weights first, then run the test command.

|                          Method                           | Backbone |     Test set     | SCORE |
| :-------------------------------------------------------: | :------: | :--------------: | :---: |
| [uninext_r50](./configs/uninext_r50-8e_youtubevis2019.py) | ResNet50 | youtube_vis_2019 | 53.3  |

## Citation

```BibTeX
@inproceedings{UNINEXT,
  title={Universal Instance Perception as Object Discovery and Retrieval},
  author={Yan, Bin and Jiang, Yi and Wu, Jiannan and Wang, Dong and Yuan, Zehuan and Luo, Ping and Lu, Huchuan},
  booktitle={CVPR},
  year={2023}
}
```

## Checklist

<!-- Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress. The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.
OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.
Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmdet.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Test-time correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

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

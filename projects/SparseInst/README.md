<div align="center">
<img src="https://github.com/hustvl/SparseInst/raw/main/assets/banner.gif">
<br>
<br>
Tianheng Cheng, <a href="https://xinggangw.info/">Xinggang Wang</a><sup><span>&#8224;</span></sup>, Shaoyu Chen, Wenqiang Zhang, <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN">Qian Zhang</a>, <a href="https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN">Chang Huang</a>, <a href="https://zhaoxiangzhang.net/">Zhaoxiang Zhang</a>, <a href="http://eic.hust.edu.cn/professor/liuwenyu/"> Wenyu Liu</a>
</br>
(<span>&#8224;</span>: corresponding author)
<div>
<a href="https://arxiv.org/abs/2203.12827">[arXiv paper]</a>
<a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Sparse_Instance_Activation_for_Real-Time_Instance_Segmentation_CVPR_2022_paper.pdf">[CVPR paper]</a>
<a href="https://drive.google.com/file/d/1xhqQvQ0YVCHd8XQxnCVqef75Hey7kI-d/view?usp=sharing">[slides]</a>
</div>
</div>

## Description

This is an implementation of [SparseInst](https://github.com/hustvl/SparseInst) based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

**SparseInst** is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation.
In contrast to region boxes or anchors (centers), SparseInst adopts a sparse set of **instance activation maps** as object representation, to highlight informative regions for each foreground objects.
Then it obtains the instance-level features by aggregating features according to the highlighted regions for recognition and segmentation.
The bipartite matching compels the instance activation maps to predict objects in a one-to-one style, thus avoiding non-maximum suppression (NMS) in post-processing. Owing to the simple yet effective designs with instance activation maps, SparseInst has extremely fast inference speed and achieves **40 FPS** and **37.9 AP** on COCO (NVIDIA 2080Ti), significantly outperforms the counter parts in terms of speed and accuracy.

<center>
<img src="https://github.com/hustvl/SparseInst/raw/main/assets/sparseinst.png">
</center>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/SparseInst/configs/sparseinst_r50_iam_8xb8-ms-270k_coco.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/SparseInst/configs/sparseinst_r50_iam_8xb8-ms-270k_coco.py
```

### Testing commands

In MMDetection's root directory, run the following command to test the model:

```bash
python tools/test.py projects/SparseInst/configs/sparseinst_r50_iam_8xb8-ms-270k_coco.py ${CHECKPOINT_PATH}
```

## Results

Here we provide the baseline version of SparseInst with ResNet50 backbone.

To find more variants, please visit the [official model zoo](https://github.com/hustvl/SparseInst#models).

| Backbone |  Style  | Lr schd | Mem (GB) | FPS  | mask AP val2017 |                           Config                            |                                                                                                                                                                    Download                                                                                                                                                                    |
| :------: | :-----: | :-----: | :------: | :--: | :-------------: | :---------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | PyTorch |  270k   |   8.7    | 44.3 |      32.9       | [config](./configs/sparseinst_r50_iam_8xb8-ms-270k_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/sparseinst/sparseinst_r50_iam_8xb8-ms-270k_coco/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/sparseinst/sparseinst_r50_iam_8xb8-ms-270k_coco/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051.json) |

## Citation

If you find SparseInst is useful in your research or applications, please consider giving a star 🌟 to the [official repository](https://github.com/hustvl/SparseInst) and citing SparseInst by the following BibTeX entry.

```BibTeX
@inproceedings{Cheng2022SparseInst,
  title     =   {Sparse Instance Activation for Real-Time Instance Segmentation},
  author    =   {Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu},
  booktitle =   {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2022}
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

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/mmdet/datasets/transforms/transforms.py#L41-L169) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmdetection/blob/5b0d5b40d5c6cfda906db7464ca22cbd4396728a/tests/test_datasets/test_transforms/test_transforms.py#L35-L88) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] Metafile.yml

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.

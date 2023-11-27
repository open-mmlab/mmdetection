# Ovdet

> [Aligning Bag of Regions for Open-Vocabulary Object Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Aligning_Bag_of_Regions_for_Open-Vocabulary_Object_Detection_CVPR_2023_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

Pre-trained vision-language models (VLMs) learn to
align vision and language representations on large-scale
datasets, where each image-text pair usually contains a bag
of semantic concepts. However, existing open-vocabulary
object detectors only align region embeddings individually
with the corresponding features extracted from the VLMs.
Such a design leaves the compositional structure of semantic
concepts in a scene under-exploited, although the structure
may be implicitly learned by the VLMs. In this work, we
propose to align the embedding of bag of regions beyond individual regions. The proposed method groups contextually
interrelated regions as a bag. The embeddings of regions
in a bag are treated as embeddings of words in a sentence,
and they are sent to the text encoder of a VLM to obtain the
bag-of-regions embedding, which is learned to be aligned
to the corresponding features extracted by a frozen VLM.
Applied to the commonly used Faster R-CNN, our approach
surpasses the previous best results by 4.6 box AP50 and 2.8
mask AP on novel categories of open-vocabulary COCO
and LVIS benchmarks, respectively. Code and models are
available at https://github.com/wusize/ovdet

### Obtain Checkpoints

#### CLIP

We use CLIP's ViT-B-32 model for the implementation of our method. Obtain the state_dict
of the model from [GoogleDrive](https://drive.google.com/file/d/1ilxBhjb3JXNDar8lKRQ9GA4hTmjxADfu/view?usp=sharing) and
put it under `checkpoints`. Otherwise, `pip install git+https://github.com/openai/CLIP.git` and
run

```python
import clip
import torch
model, _ = clip.load("ViT-B/32")
torch.save(model.state_dict(), 'checkpoints/clip_vitb32.pth')
```

#### FasterRCNN+ResNet50+FPN

Train the detector based on FasterRCNN+ResNet50+FPN with SyncBN and SOCO pre-trained model. Obtain the SOCO pre-trained
model from [GoogleDrive](https://drive.google.com/file/d/1rIW9IXjWEnFZa4klZuZ5WNSchRYaOC0x/view?usp=sharing) and put it
under `checkpoints`.

### Install Extra Package

```bash
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install ftfy
pip install regex
```

## Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection).
Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and put them
under `data/coco/wusize`
The data structure looks like:

```text
checkpoints/
├── clip_vitb32.pth
├── res50_fpn_soco_star_400.pth
data/
├── coco
│   ├── annotations
│   │   ├── instances_{train,val}2017.json
│   ├── wusize
│   │   ├── instances_train2017_base.json
│   │   ├── instances_val2017_base.json
│   │   ├── instances_val2017_novel.json
│   │   ├── captions_train2017_tags_allcaps.json
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

The json file for caption supervision `captions_train2017_tags_allcaps.json` is obtained following
[Detic](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#:~:text=Next%2C%20we%20preprocess%20the%20COCO%20caption%20data%3A). Put it under
`data/coco/wusize`.

### Inference

If you just want to run inference, you can try in [inference.ipynb](inference.ipynb)

### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/Ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py
```

For multi-gpu training, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh projects/Ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py 4
```

### Testing commands

In MMDetection's root directory, run the following command to test the model:

```bash
python tools/test.py projects/Ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py ${CHECKPOINT_PATH}
```

## Results

he implementation based on MMDet3.x achieves better results compared to the results reported in the paper.

|           | Backbone | Method | Supervision  | Novel AP50 |                                    Config                                     |                                                                                           Download                                                                                            |
| :-------: | :------: | :----: | :----------: | :--------: | :---------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   Paper   | R-50-FPN | BARON  |     CLIP     |    34.0    |                                       -                                       |                                                                                               -                                                                                               |
| This Repo | R-50-FPN | BARON  |     CLIP     |    34.6    | [config](configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py)  | [model](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1JTM0uoPQZtq7lnhZxCBwjxBUca9omYR9?usp=sharing) |
|   Paper   | R-50-C4  | BARON  | COCO Caption |    33.1    |                                       -                                       |                                                                                               -                                                                                               |
| This Repo | R-50-C4  | BARON  | COCO Caption |    35.1    | [config](configs/baron/ov_coco/baron_caption_faster_rcnn_r50_caffe_c4_90k.py) | [model](https://drive.google.com/drive/folders/1b-ueEz57alju9qamADm7BmDCaL-NWnSn?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1b-ueEz57alju9qamADm7BmDCaL-NWnSn?usp=sharing) |
| This Repo | R-50-C4  | BARON  |     CLIP     |    34.0    |   [config](configs/baron/ov_coco/baron_kd_faster_rcnn_r50_caffe_c4_90k.py)    | [model](https://drive.google.com/drive/folders/1ckS8Cju2xQyHfxMsQRPd5h7qKhwlWOyV?usp=sharing) \|  [log](https://drive.google.com/drive/folders/1ckS8Cju2xQyHfxMsQRPd5h7qKhwlWOyV?usp=sharing) |

## Citation

<!-- You may remove this section if not applicable. -->

```latex
@inproceedings{wu2023baron,
    title={Aligning Bag of Regions for Open-Vocabulary Object Detection},
    author={Size Wu and Wenwei Zhang and Sheng Jin and Wentao Liu and Chen Change Loy},
    year={2023},
    booktitle={CVPR},
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

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/metafile.yml) -->

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

  <!-- In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/README.md) -->

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.

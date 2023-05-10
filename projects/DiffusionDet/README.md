## Description

This is an implementation of [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet) based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main), [MMCV](https://github.com/open-mmlab/mmcv), and [MMEngine](https://github.com/open-mmlab/mmengine).

<center>
<img src="https://user-images.githubusercontent.com/48282753/211472911-c84d658a-952b-4608-8b91-9ac932cbf2e2.png">
</center>

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Comparison of results

1. Download the [DiffusionDet released model](https://github.com/ShoufaChen/DiffusionDet#models).

2. Convert model from DiffusionDet version to MMDetection version. We give a [sample script](model_converters/diffusiondet_resnet_to_mmdet.py)
   to convert `DiffusionDet-resnet50` model. Users can download the corresponding models from [here](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth).

   ```shell
   python projects/DiffusionDet/model_converters/diffusiondet_resnet_to_mmdet.py ${DiffusionDet ckpt path} ${MMDetectron ckpt path}
   ```

3. Testing the model in MMDetection.

   ```shell
   python tools/test.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py ${CHECKPOINT_PATH}
   ```

**Note:** During inference time, DiffusionDet will randomly generate noisy boxes,
which may affect the AP results. If users want to get the same result every inference time, setting seed is a good way.
We give a table to compare the inference results on `ResNet50-500-proposals` between DiffusionDet and MMDetection.

|                                                         Config                                                          | Step |    AP     |
| :---------------------------------------------------------------------------------------------------------------------: | :--: | :-------: |
| [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet/blob/main/configs/diffdet.coco.res50.yaml) (released results) |  1   |   45.5    |
|      [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet/blob/main/configs/diffdet.coco.res50.yaml) (seed=0)      |  1   |   45.66   |
|         [MMDetection](configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py) (seed=0)          |  1   |   45.7    |
|       [MMDetection](configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py) (random seed)       |  1   | 45.6~45.8 |
| [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet/blob/main/configs/diffdet.coco.res50.yaml) (released results) |  4   |   46.1    |
|      [DiffusionDet](https://github.com/ShoufaChen/DiffusionDet/blob/main/configs/diffdet.coco.res50.yaml) (seed=0)      |  4   |   46.38   |
|         [MMDetection](configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py) (seed=0)          |  4   |   46.4    |
|       [MMDetection](configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py) (random seed)       |  4   | 46.2~46.4 |

- `seed=0` means hard set seed before generating random boxes.
  ```python
  # hard set seed=0 before generating random boxes
  seed = 0
  random.seed(seed)
  torch.manual_seed(seed)
  # torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  ...
  noise_bboxes_raw = torch.randn(
      (self.num_proposals, 4),
      device=device)
  ...
  ```
- `random seed` means do not hard set seed before generating random boxes.

### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py
```

For multi-gpu training, run:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py
```

### Testing commands

In MMDetection's root directory, run the following command to test the model:

```bash
# for 1 step inference
# test command
python tools/test.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py ${CHECKPOINT_PATH}

# for 4 steps inference

# test command
python tools/test.py projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py ${CHECKPOINT_PATH} --cfg-options model.bbox_head.sampling_timesteps=4
```

**Note:** There is no difference between 1 step or 4 steps (or other multi-step) during training. Users can set different steps during inference through `--cfg-options model.bbox_head.sampling_timesteps=${STEPS}`, but larger `sampling_timesteps` will affect the inference time.

## Results

Here we provide the baseline version of DiffusionDet with ResNet50 backbone.

To find more variants, please visit the [official model zoo](https://github.com/ShoufaChen/DiffusionDet#models).

| Backbone |  Style  | Lr schd | AP (Step=1) | AP (Step=4) |                                           Config                                           |                                                                                                                                                                                                                                      Download                                                                                                                                                                                                                                      |
| :------: | :-----: | :-----: | :---------: | :---------: | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | PyTorch |  450k   |    44.5     |    46.2     | [config](./configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/diffusiondet/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_20230215_090925-7d6ed504.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/diffusiondet/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_20230215_090925.log.json) |

## License

DiffusionDet is under the [CC-BY-NC 4.0 license](https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE). Users should be careful about adopting these features in any commercial matters.

## Citation

If you find DiffusionDet is useful in your research or applications, please consider giving a star 🌟 to the [official repository](https://github.com/ShoufaChen/DiffusionDet) and citing DiffusionDet by the following BibTeX entry.

```BibTeX
@article{chen2022diffusiondet,
      title={DiffusionDet: Diffusion Model for Object Detection},
      author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
      journal={arXiv preprint arXiv:2211.09788},
      year={2022}
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

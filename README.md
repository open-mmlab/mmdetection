
# mmdetection

## Introduction

The master branch works with **PyTorch 1.0**. If you would like to use PyTorch 0.4.1,
please checkout to the [pytorch-0.4.1](https://github.com/open-mmlab/mmdetection/tree/pytorch-0.4.1) branch.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)

### Major features

- **Modular Design**

  One can easily construct a customized object detection framework by combining different components.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **Efficient**

  All basic bbox and mask operations run on GPUs now.
  The training speed is about 5% ~ 20% faster than Detectron for different models.

- **State of the art**

  This was the codebase of the *MMDet* team, who won the [COCO Detection 2018 challenge](http://cocodataset.org/#detection-leaderboard).

Apart from mmdetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research,
which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Updates

v0.6.0 (14/04/2019)
- Up to 30% speedup compared to the model zoo.
- Support both PyTorch stable and nightly version.
- Replace NMS and SigmoidFocalLoss with Pytorch CUDA extensions.

v0.6rc0(06/02/2019)
- Migrate to PyTorch 1.0.

v0.5.7 (06/02/2019)
- Add support for Deformable ConvNet v2. (Many thanks to the authors and [@chengdazhi](https://github.com/chengdazhi))
- This is the last release based on PyTorch 0.4.1.

v0.5.6 (17/01/2019)
- Add support for Group Normalization.
- Unify RPNHead and single stage heads (RetinaHead, SSDHead) with AnchorHead.

v0.5.5 (22/12/2018)
- Add SSD for COCO and PASCAL VOC.
- Add ResNeXt backbones and detection models.
- Refactoring for Samplers/Assigners and add OHEM.
- Add VOC dataset and evaluation scripts.

v0.5.4 (27/11/2018)
- Add SingleStageDetector and RetinaNet.

v0.5.3 (26/11/2018)
- Add Cascade R-CNN and Cascade Mask R-CNN.
- Add support for Soft-NMS in config files.

v0.5.2 (21/10/2018)
- Add support for custom datasets.
- Add a script to convert PASCAL VOC annotations to the expected format.

v0.5.1 (20/10/2018)
- Add BBoxAssigner and BBoxSampler, the `train_cfg` field in config files are restructured.
- `ConvFCRoIHead` / `SharedFCRoIHead` are renamed to `ConvFCBBoxHead` / `SharedFCBBoxHead` for consistency.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      |
|--------------------|:--------:|:--------:|:--------:|:--------:|
| RPN                | ✓        | ✓        | ☐        | ✗        |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        |
| SSD                | ✗        | ✗        | ✗        | ✓        |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        |

Other features
- [x] DCNv2
- [x] Group Normalization
- [x] OHEM
- [x] Soft-NMS


## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Inference with pretrained models

### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize detection results

We allow to run one or multiple processes on each GPU, e.g. 8 processes on 8 GPU
or 16 processes on 8 GPU. When the GPU workload is not very heavy for a single
process, running multiple processes will accelerate the testing, which is specified
with the argument `--proc_per_gpu <PROCESS_NUM>`.


To test a dataset and save the results.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE>
```

To perform evaluation after testing, add `--eval <EVAL_TYPES>`. Supported types are:
`[proposal_fast, proposal, bbox, segm, keypoints]`.
`proposal_fast` denotes evaluating proposal recalls with our own implementation,
others denote evaluating the corresponding metric with the official coco api.

For example, to evaluate Mask R-CNN with 8 GPUs and save the result as `results.pkl`.

```shell
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py <CHECKPOINT_FILE> --gpus 8 --out results.pkl --eval bbox segm
```

It is also convenient to visualize the results during testing by adding an argument `--show`.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --show
```

### Test image(s)

We provide some high-level apis (experimental) to test an image.

```python
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
img = mmcv.imread('test.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result)

# test a list of images
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
```


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

### Distributed training (Single or Multiples machines)

mmdetection potentially supports multiple launch methods, e.g., PyTorch’s built-in launch utility, slurm and MPI.

We provide a training script using the launch utility provided by PyTorch.

```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be replaced.

Expected results in WORK_DIR:

- log file
- saved checkpoints (every k epochs, defaults=1)
- a symbol link to the latest checkpoint

**Important**: The default learning rate is for 8 GPUs. If you use less or more than 8 GPUs, you need to set the learning rate proportional to the GPU num. E.g., modify lr to 0.01 for 4 GPUs or 0.04 for 16 GPUs.

### Non-distributed training

Please refer to `tools/train.py` for non-distributed training, which is not recommended
and left for debugging. Even on a single machine, distributed training is preferred.

### Train on custom datasets

We define a simple annotation format.

The annotation of a dataset is a list of dict, each dict corresponds to an image.
There are 3 field `filename` (relative path), `width`, `height` for testing,
and an additional field `ann` for training. `ann` is also a dict containing at least 2 fields:
`bboxes` and `labels`, both of which are numpy arrays. Some datasets may provide
annotations like crowd/difficult/ignored bboxes, we use `bboxes_ignore` and `labels_ignore`
to cover them.

Here is an example.
```
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray> (n, 4),
            'labels': <np.ndarray> (n, ),
            'bboxes_ignore': <np.ndarray> (k, 4),
            'labels_ignore': <np.ndarray> (k, ) (optional field)
        }
    },
    ...
]
```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from `CustomDataset`, and overwrite two methods
  `load_annotations(self, ann_file)` and `get_ann_info(self, idx)`, like [CocoDataset](mmdet/datasets/coco.py) and [VOCDataset](mmdet/datasets/voc.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle or json file, like [pascal_voc.py](tools/convert_datasets/pascal_voc.py).
  Then you can simply use `CustomDataset`.

## Technical details

Some implementation details and project structures are described in the [technical details](TECHNICAL_DETAILS.md).

## Citation

If you use our codebase or models in your research, please cite this project.
We will release a paper or technical report later.

```
@misc{mmdetection2018,
  author =       {Kai Chen and Jiangmiao Pang and Jiaqi Wang and Yu Xiong and Xiaoxiao Li
                  and Shuyang Sun and Wansen Feng and Ziwei Liu and Jianping Shi and
                  Wanli Ouyang and Chen Change Loy and Dahua Lin},
  title =        {mmdetection},
  howpublished = {\url{https://github.com/open-mmlab/mmdetection}},
  year =         {2018}
}
```

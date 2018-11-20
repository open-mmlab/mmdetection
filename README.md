
# mmdetection

## Introduction

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

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

v0.5.2 (21/10/2018)
- Add support for custom datasets.
- Add a script to convert PASCAL VOC annotations to the expected format.

v0.5.1 (20/10/2018)
- Add BBoxAssigner and BBoxSampler, the `train_cfg` field in config files are restructured.
- `ConvFCRoIHead` / `SharedFCRoIHead` are renamed to `ConvFCBBoxHead` / `SharedFCBBoxHead` for consistency.

## Benchmark and model zoo

We provide our baseline results and the comparision with Detectron, the most
popular detection projects. Results and models are available in the [Model zoo](MODEL_ZOO.md).

## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 3.4+
- PyTorch 0.4.1 and torchvision
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install mmdetection

a. Install PyTorch 0.4.1 and torchvision following the [official instructions](https://pytorch.org/).

b. Clone the mmdetection repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
```

c. Compile cuda extensions.

```shell
cd mmdetection
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

d. Install mmdetection (other dependencies will be installed automatically).

```shell
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

Note: You need to run the last step each time you pull updates from github.
The git commit id will be written to the version number and also saved in trained models.

### Prepare COCO dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017

```

> [Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda for reference.


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

- proposal_fast: eval recalls of proposals with our own codes. (supposed to get the same results as the official evaluation)
- proposal: eval recalls of proposals with the official code provided by COCO.
- bbox: eval box AP with the official code provided by COCO.
- segm: eval mask AP with the official code provided by COCO.
- keypoints: eval keypoint AP with the official code provided by COCO.

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

### Distributed training

mmdetection potentially supports multiple launch methods, e.g., PyTorch’s built-in launch utility, slurm and MPI.

We provide a training script using the launch utility provided by PyTorch.

```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be overwritten.

### Non-distributed training

```shell
python tools/train.py <CONFIG_FILE> --gpus <GPU_NUM> --work_dir <WORK_DIR> --validate
```

Expected results in WORK_DIR:

- log file
- saved checkpoints (every k epochs, defaults=1)
- a symbol link to the latest checkpoint

> **Note**
> 1. We recommend using distributed training with NCCL2 even on a single machine, which is faster. Non-distributed training is for debugging or other purposes.
> 2. The default learning rate is for 8 GPUs. If you use less or more than 8 GPUs, you need to set the learning rate proportional to the GPU num. E.g., modify lr to 0.01 for 4 GPUs or 0.04 for 16 GPUs.

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
            'labels_ignore': <np.ndarray> (k, 4) (optional field)
        }
    },
    ...
]
```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from `CustomDataset`, and overwrite two methods
  `load_annotations(self, ann_file)` and `get_ann_info(self, idx)`, like [CocoDataset](mmdet/datasets/coco.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle file, like [pascal_voc.py](tools/convert_datasets/pascal_voc.py).
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

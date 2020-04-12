# Tutorials 2: Adding New Dataset

## Customize datasets

The simplest way is to convert your dataset to existing dataset formats (COCO or PASCAL VOC).

Here we show an example of using a custom dataset of 5 classes, assuming it is also in COCO format.

In `configs/my_custom_config.py`:

```python
...
# dataset settings
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
...
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ...),
    val=dict(
        type=dataset_type,
        classes=classes,
        ...),
    test=dict(
        type=dataset_type,
        classes=classes,
        ...))
...
```

It is also fine if you do not want to convert the annotation format to COCO or PASCAL format.
Actually, we define a simple annotation format and all existing datasets are
processed to be compatible with it, either online or offline.

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
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
    },
    ...
]
```

There are two ways to work with custom datasets.

- online conversion

  You can write a new Dataset class inherited from `CustomDataset`, and overwrite two methods
  `load_annotations(self, ann_file)` and `get_ann_info(self, idx)`,
  like [CocoDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py) and [VOCDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/voc.py).

- offline conversion

  You can convert the annotation format to the expected format above and save it to
  a pickle or json file, like [pascal_voc.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/convert_datasets/pascal_voc.py).
  Then you can simply use `CustomDataset`.

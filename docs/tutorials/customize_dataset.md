# Tutorial 2: Customize Datasets

## Support new data format

To support a new data format, you can either convert them to existing formats (COCO format or PASCAL format) or directly convert them to the middle format. You could also choose to convert them offline (before training by a script) or online (implement a new dataset and do the conversion at training). In MMDetection, we recommand to convert the data into COCO formats and do the conversion offline, thus you only need to modify the config's data annotation pathes and classes after the conversion to your data.

### Reorganize new data formats to existing format

The simplest way is to convert your dataset to existing dataset formats (COCO or PASCAL VOC).

The annotation json files in COCO format has the following necessary keys:

```python
'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
```

There are three necessary keys in the json file:

- `images`: contains a list of images with their informations like `file_name`, `height`, `width`, and `id`.
- `annotations`: contains the list of instance annotations.
- `categories`: contains the list of categories names and their ID.

After the data pre-processing, the users need to further modify the config files to use the dataset.
Here we show an example of using a custom dataset of 5 classes, assuming it is also in COCO format.

In `configs/my_custom_config.py`:

```python
...
# dataset settings
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/train/data',
        ...),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/val/data',
        ...),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='path/to/your/test/data',
        ...))
...
```

We use this way to support CityScapes dataset. The script is in [cityscapes.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/convert_datasets/cityscapes.py) and we also provide the finetuning [configs](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes).

**Note**

1. For instance segmentation datasets, **MMDetection only supports evaluating mask AP of dataset in COCO format for now**.
2. It is recommanded to convert the data offline before training, thus you can still use `CocoDataset` and only need to modify the path of annotations and the training classes.

### Reorganize new data format to middle format

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

```python

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

### An example of customized dataset

Assume the annotation is in a new format in text files.
The bounding boxes annotations are stored in text file `annotation.txt` as the following

```
#
000001.jpg
1280 720
2
10 20 40 60 1
20 40 50 60 2
#
000002.jpg
1280 720
3
50 20 40 60 2
20 40 30 45 2
30 40 50 60 3
```

We can create a new dataset in `mmdet/datasets/my_dataset.py` to load the data.

```python
import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle')

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            anns = ann_line.split(' ')
            bboxes = []
            labels = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))

            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

```

Then in the config, to use `MyDataset` you can modify the config as the following

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

## Customize datasets by dataset wrappers

MMDetection also supports many dataset wrappers to mix the dataset or modify the dataset distribution for training.
Currently it supports to three dataset wrappers as below:

- `RepeatDataset`: simply repeat the whole dataset.
- `ClassBalancedDataset`: repeat dataset in a class balanced manner.
- `ConcatDataset`: concat datasets.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### Class balanced dataset

We use `ClassBalancedDataset` as wrapper to repeat the dataset based on category
frequency. The dataset to repeat needs to instantiate function `self.get_cat_ids(idx)`
to support `ClassBalancedDataset`.
For example, to repeat `Dataset_A` with `oversample_thr=1e-3`, the config looks like the following

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

You may refer to [source code](../../mmdet/datasets/dataset_wrappers.py) for details.

### Concatenate dataset

There are three ways to concatenate the dataset.

1. If the datasets you want to concatenate are in the same type with different annotation files, you can concatenate the dataset configs like the following.

    ```python
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        pipeline=train_pipeline
    )
    ```

    If the concatenated dataset is used for test or evaluation, this manner supports to evaluate each dataset separately. To test the concatenated datasets as a whole, you can set `separate_eval=False` as below.

    ```python
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        separate_eval=False,
        pipeline=train_pipeline
    )
    ```

2. In case the dataset you want to concatenate is different, you can concatenate the dataset configs like the following.

    ```python
    dataset_A_train = dict()
    dataset_B_train = dict()

    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train = [
            dataset_A_train,
            dataset_B_train
        ],
        val = dataset_A_val,
        test = dataset_A_test
        )
    ```

    If the concatenated dataset is used for test or evaluation, this manner also supports to evaluate each dataset separately.

3. We also support to define `ConcatDataset` explicitly as the following.

    ```python
    dataset_A_val = dict()
    dataset_B_val = dict()

    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dataset_A_train,
        val=dict(
            type='ConcatDataset',
            datasets=[dataset_A_val, dataset_B_val],
            separate_eval=False))
    ```

    This manner allows users to evaluate all the datasets as a single one by setting `separate_eval=False`.

**Note:**

1. The option `separate_eval=False` assumes the datasets use `self.data_infos` during evaluation. Therefore, COCO datasets do not support this behavior since COCO datasets do not fully rely on `self.data_infos` for evaluation. Combining different types of datasets and evaluating them as a whole is not tested thus is not suggested.
2. Evaluating `ClassBalancedDataset` and `RepeatDataset` is not supported thus evaluating concatenated datasets of these types is also not supported.

A more complex example that repeats `Dataset_A` and `Dataset_B` by N and M times, respectively, and then concatenates the repeated datasets is as the following.

```python
dataset_A_train = dict(
    type='RepeatDataset',
    times=N,
    dataset=dict(
        type='Dataset_A',
        ...
        pipeline=train_pipeline
    )
)
dataset_A_val = dict(
    ...
    pipeline=test_pipeline
)
dataset_A_test = dict(
    ...
    pipeline=test_pipeline
)
dataset_B_train = dict(
    type='RepeatDataset',
    times=M,
    dataset=dict(
        type='Dataset_B',
        ...
        pipeline=train_pipeline
    )
)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train = [
        dataset_A_train,
        dataset_B_train
    ],
    val = dataset_A_val,
    test = dataset_A_test
)

```

## Modify Dataset Classes

With existing dataset types, we can modify the class names of them to train subset of the annotations.
For example, if you want to train only three classes of the current dataset,
you can modify the classes of dataset.
The dataset will filter out the ground truth boxes of other classes automatically.

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMDetection V2.0 also supports to read the classes from a file, which is common in real applications.
For example, assume the `classes.txt` contains the name of classes as the following.

```
person
bicycle
car
```

Users can set the classes as a file path, the dataset will load it and convert it to a list automatically.

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

**Note**:

- Before MMDetection v2.5.0, the dataset will filter out the empty GT images automatically if the classes are set and there is no way to disable that through config. This is an undesirable behavior and introduces confusion because if the classes are not set, the dataset only filter the empty GT images when `filter_empty_gt=True` and `test_mode=False`. After MMDetection v2.5.0, we decouple the image filtering process and the classes modification, i.e., the dataset will only filter empty GT images when `filter_empty_gt=True` and `test_mode=False`, no matter whether the classes are set. Thus, setting the classes only influences the annotations of classes used for training and users could decide whether to filter empty GT images by themselves.
- Since the middle format only has box labels and does not contain the class names, when using `CustomDataset`, users cannot filter out the empty GT images through configs but only do this offline.
- The features for setting dataset classes and dataset filtering will be refactored to be more user-friendly in v2.6.0 or v2.7.0 (depends on the progress).

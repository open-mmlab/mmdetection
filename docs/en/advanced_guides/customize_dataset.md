# Customize Datasets

## Support new data format

To support a new data format, you can either convert them to existing formats (COCO format or PASCAL format) or directly convert them to the middle format. You could also choose to convert them offline (before training by a script) or online (implement a new dataset and do the conversion at training). In MMDetection, we recommend to convert the data into COCO formats and do the conversion offline, thus you only need to modify the config's data annotation paths and classes after the conversion of your data.

### Reorganize new data formats to existing format

The simplest way is to convert your dataset to existing dataset formats (COCO or PASCAL VOC).

The annotation JSON files in COCO format has the following necessary keys:

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
            249.06]],  # If you have mask labels, and it is in polygon XY point coordinate format, you need to ensure that at least 3 point coordinates are included. Otherwise, it is an invalid polygon.
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

There are three necessary keys in the JSON file:

- `images`: contains a list of images with their information like `file_name`, `height`, `width`, and `id`.
- `annotations`: contains the list of instance annotations.
- `categories`: contains the list of categories names and their ID.

After the data pre-processing, there are two steps for users to train the customized new dataset with existing format (e.g. COCO format):

1. Modify the config file for using the customized dataset.
2. Check the annotations of the customized dataset.

Here we give an example to show the above two steps, which uses a customized dataset of 5 classes with COCO format to train an existing Cascade Mask R-CNN R50-FPN detector.

#### 1. Modify the config file for using the customized dataset

There are two aspects involved in the modification of config file:

1. The `data` field. Specifically, you need to explicitly add the `metainfo=dict(classes=classes)` fields in `train_dataloader.dataset`, `val_dataloader.dataset` and `test_dataloader.dataset` and `classes` must be a tuple type.
2. The `num_classes` field in the `model` part. Explicitly over-write all the `num_classes` from default value (e.g. 80 in COCO) to your classes number.

In `configs/my_custom_config.py`:

```python

# the new config inherits the base configs to highlight the necessary modification
_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
data_root='path/to/your/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/annotation_data',
        data_prefix=dict(img='train/image_data')
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotation_data',
        data_prefix=dict(img='val/image_data')
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='test/image_data')
        )
    )

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5)],
    # explicitly over-write all the `num_classes` field from default 80 to 5.
    mask_head=dict(num_classes=5)))
```

#### 2. Check the annotations of the customized dataset

Assuming your customized dataset is COCO format, make sure you have the correct annotations in the customized dataset:

1. The length for `categories` field in annotations should exactly equal the tuple length of `classes` fields in your config, meaning the number of classes (e.g. 5 in this example).
2. The `classes` fields in your config file should have exactly the same elements and the same order with the `name` in `categories` of annotations. MMDetection automatically maps the uncontinuous `id` in `categories` to the continuous label indices, so the string order of `name` in `categories` field affects the order of label indices. Meanwhile, the string order of `classes` in config affects the label text during visualization of predicted bounding boxes.
3. The `category_id` in `annotations` field should be valid, i.e., all values in `category_id` should belong to `id` in `categories`.

Here is a valid example of annotations:

```python

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

# MMDetection automatically maps the uncontinuous `id` to the continuous label indices.
'categories': [
    {'id': 1, 'name': 'a'}, {'id': 3, 'name': 'b'}, {'id': 4, 'name': 'c'}, {'id': 16, 'name': 'd'}, {'id': 17, 'name': 'e'},
 ]
```

We use this way to support CityScapes dataset. The script is in [cityscapes.py](../../../tools/dataset_converters/cityscapes.py) and we also provide the finetuning [configs](../../../configs/cityscapes).

**Note**

1. For instance segmentation datasets, **MMDetection only supports evaluating mask AP of dataset in COCO format for now**.
2. It is recommended to convert the data offline before training, thus you can still use `CocoDataset` and only need to modify the path of annotations and the training classes.

### Reorganize new data format to middle format

It is also fine if you do not want to convert the annotation format to COCO or PASCAL format.
Actually, we define a simple annotation format in MMEninge's [BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L116) and all existing datasets are
processed to be compatible with it, either online or offline.

The annotation of the dataset must be in `json` or `yaml`, `yml` or `pickle`, `pkl` format; the dictionary stored in the annotation file must contain two fields `metainfo` and `data_list`.  The `metainfo` is a dictionary, which contains the metadata of the dataset, such as class information; `data_list` is a list, each element in the list is a dictionary, the dictionary defines the raw data of one image, and each raw data contains a or several training/testing samples.

Here is an example.

```python
{
    'metainfo':
        {
            'classes': ('person', 'bicycle', 'car', 'motorcycle'),
            ...
        },
    'data_list':
        [
            {
                "img_path": "xxx/xxx_1.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "ignore_flag": 0
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "ignore_flag": 0
                  }
                ]
              },
            {
                "img_path": "xxx/xxx_2.jpg",
                "height": 320,
                "width": 460,
                "instances":
                [
                  {
                    "bbox": [10, 0, 20, 20],
                    "bbox_label": 3,
                    "ignore_flag": 1,
                  }
                ]
              },
            ...
        ]
}
```

Some datasets may provide annotations like crowd/difficult/ignored bboxes, we use `ignore_flag`to cover them.

After obtaining the above standard data annotation format, you can directly use [BaseDetDataset](../../../mmdet/datasets/base_det_dataset.py#L13) of MMDetection in the configuration , without conversion.

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
import mmengine

from mmdet.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDetDataset):

    METAINFO = {
       'classes': ('person', 'bicycle', 'car', 'motorcycle'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

    def load_data_list(self, ann_file):
        ann_list = mmengine.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            instances = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                instance = {}
                instance['bbox'] = [float(ann) for ann in anns.split(' ')[:4]]
                instance['bbox_label']=int(anns[4])
 				instances.append(instance)

            data_infos.append(
                dict(
                    img_path=ann_list[i + 1],
                    img_id=i,
                    width=width,
                    height=height,
                    instances=instances
                ))

        return data_infos
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

MMEngine also supports many dataset wrappers to mix the dataset or modify the dataset distribution for training.
Currently it supports to three dataset wrappers as below:

- `RepeatDataset`: simply repeat the whole dataset.
- `ClassBalancedDataset`: repeat dataset in a class balanced manner.
- `ConcatDataset`: concat datasets.

For detailed usage, see [MMEngine Dataset Wrapper](#TODO).

## Modify Dataset Classes

With existing dataset types, we can modify the metainfo of them to train subset of the annotations.
For example, if you want to train only three classes of the current dataset,
you can modify the classes of dataset.
The dataset will filter out the ground truth boxes of other classes automatically.

```python
classes = ('person', 'bicycle', 'car')
train_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
val_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
test_dataloader = dict(
    dataset=dict(
        metainfo=dict(classes=classes))
    )
```

**Note**:

- Before MMDetection v2.5.0, the dataset will filter out the empty GT images automatically if the classes are set and there is no way to disable that through config. This is an undesirable behavior and introduces confusion because if the classes are not set, the dataset only filter the empty GT images when `filter_empty_gt=True` and `test_mode=False`. After MMDetection v2.5.0, we decouple the image filtering process and the classes modification, i.e., the dataset will only filter empty GT images when `filter_cfg=dict(filter_empty_gt=True)` and `test_mode=False`, no matter whether the classes are set. Thus, setting the classes only influences the annotations of classes used for training and users could decide whether to filter empty GT images by themselves.
- When directly using `BaseDataset` in MMEngine or `BaseDetDataset` in MMDetection, users cannot filter images without GT by modifying the configuration, but it can be solved in an offline way.
- Please remember to modify the `num_classes` in the head when specifying `classes` in dataset. We implemented [NumClassCheckHook](../../../mmdet/engine/hooks/num_class_check_hook.py) to check whether the numbers are consistent since v2.9.0(after PR#4508).

## COCO Panoptic Dataset

Now we support COCO Panoptic Dataset, the format of panoptic annotations is different from COCO format.
Both the foreground and the background will exist in the annotation file.
The annotation json files in COCO Panoptic format has the following necessary keys:

```python
'images': [
    {
        'file_name': '000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
]

'annotations': [
    {
        'filename': '000000001268.jpg',
        'image_id': 1268,
        'segments_info': [
            {
                'id':8345037,  # One-to-one correspondence with the id in the annotation map.
                'category_id': 51,
                'iscrowd': 0,
                'bbox': (x1, y1, w, h),  # The bbox of the background is the outer rectangle of its mask.
                'area': 24315
            },
            ...
        ]
    },
    ...
]

'categories': [  # including both foreground categories and background categories
    {'id': 0, 'name': 'person'},
    ...
 ]
```

Moreover, the `seg` must be set to the path of the panoptic annotation images.

```python
dataset_type = 'CocoPanopticDataset'
data_root='path/to/your/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img='train/image_data/', seg='train/panoptic/image_annotation_data/')
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img='val/image_data/', seg='val/panoptic/image_annotation_data/')
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img='test/image_data/', seg='test/panoptic/image_annotation_data/')
    )
)
```

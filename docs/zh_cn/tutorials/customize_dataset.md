# 教程 2: 自定义数据集

## 支持新的数据格式

为了支持新的数据格式，可以选择将数据转换成现成的格式（COCO 或者 PASCAL）或将其转换成中间格式。当然也可以选择以离线的形式（在训练之前使用脚本转换）或者在线的形式（实现一个新的 dataset 在训练中进行转换）来转换数据。

在 MMDetection 中，建议将数据转换成 COCO 格式并以离线的方式进行，因此在完成数据转换后只需修改配置文件中的标注数据的路径和类别即可。

### 将新的数据格式转换为现有的数据格式

最简单的方法就是将你的数据集转换成现有的数据格式（COCO 或者 PASCAL VOC）

COCO 格式的 json 标注文件有如下必要的字段：

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
            249.06]],  # 如果有 mask 标签
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

在 json 文件中有三个必要的键：

- `images`: 包含多个图片以及它们的信息的数组，例如 `file_name`、`height`、`width` 和 `id`。
- `annotations`: 包含多个实例标注信息的数组。
- `categories`: 包含多个类别名字和 ID 的数组。

在数据预处理之后，使用现有的数据格式来训练自定义的新数据集有如下两步（以 COCO 为例）：

1. 为自定义数据集修改配置文件。
2. 检查自定义数据集的标注。

这里我们举一个例子来展示上面的两个步骤，这个例子使用包括 5 个类别的 COCO 格式的数据集来训练一个现有的 Cascade Mask R-CNN R50-FPN 检测器

#### 1. 为自定义数据集修改配置文件

配置文件的修改涉及两个方面：

1. `data` 部分。需要在 `data.train`、`data.val` 和 `data.test` 中添加 `classes`。
2. `model` 部分中的 `num_classes`。需要将默认值（COCO 数据集中为 80）修改为自定义数据集中的类别数。

`configs/my_custom_config.py` 内容如下：

```python

# 新的配置来自基础的配置以更好地说明需要修改的地方
_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 1. 数据集设定
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='path/to/your/train/annotation_data',
        img_prefix='path/to/your/train/image_data'),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='path/to/your/val/annotation_data',
        img_prefix='path/to/your/val/image_data'),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='path/to/your/test/annotation_data',
        img_prefix='path/to/your/test/image_data'))

# 2. 模型设置

# 将所有的 `num_classes` 默认值修改为5（原来为80）
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
                num_classes=5)],
    # 将所有的 `num_classes` 默认值修改为 5（原来为 80）
    mask_head=dict(num_classes=5)))
```

#### 2. 检查自定义数据集的标注

假设你自己的数据集是 COCO 格式，那么需要保证数据的标注没有问题：

1. 标注文件中 `categories` 的长度要与配置中的 `classes` 元组长度相匹配，它们都表示有几类。（如例子中有 5 个类别）
2. 配置文件中 `classes` 字段应与标注文件里 `categories` 下的 `name` 有相同的元素且顺序一致。MMDetection 会自动将 `categories` 中不连续的 `id` 映射成连续的索引，因此 `categories` 下的 `name`的字符串顺序会影响标签的索引。同时，配置文件中的 `classes` 的字符串顺序也会影响到预测框可视化时的标签。
3. `annotations` 中的 `category_id` 必须是有效的值。比如所有 `category_id` 的值都应该属于 `categories` 中的 `id`。

下面是一个有效标注的例子：

```python

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  #如果有 mask 标签。
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

# MMDetection 会自动将 `categories` 中不连续的 `id` 映射成连续的索引。
'categories': [
    {'id': 1, 'name': 'a'}, {'id': 3, 'name': 'b'}, {'id': 4, 'name': 'c'}, {'id': 16, 'name': 'd'}, {'id': 17, 'name': 'e'},
 ]
```

我们使用这种方式来支持 CityScapes 数据集。脚本在[cityscapes.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/cityscapes.py) 并且我们提供了微调的[configs](https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes).

**注意**

1. 对于实例分割数据集, **MMDetection 目前只支持评估 COCO 格式的 mask AP**.
2. 推荐训练之前进行离线转换，这样就可以继续使用 `CocoDataset` 且只需修改标注文件的路径以及训练的种类。

### 调整新的数据格式为中间格式

如果不想将标注格式转换为 COCO 或者 PASCAL 格式也是可行的。实际上，我们定义了一种简单的标注格式并且与所有现有的数据格式兼容，也能进行离线或者在线转换。

数据集的标注是包含多个字典（dict）的列表，每个字典（dict）都与一张图片对应。测试时需要用到 `filename`（相对路径）、`width` 和 `height` 三个字段；训练时则额外需要 `ann`。`ann` 也是至少包含了两个字段的字典：`bboxes` 和 `labels`，它们都是 numpy array。有些数据集可能会提供如：crowd/difficult/ignored bboxes 标注，那么我们使用 `bboxes_ignore` 以及 `labels_ignore` 来包含它们。

下面给出一个例子。

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
            'labels_ignore': <np.ndarray, int64> (k, ) （可选字段）
        }
    },
    ...
]
```

有两种方法处理自定义数据。

- 在线转换（online conversion）

  可以新写一个继承自 `CustomDataset` 的 Dataset 类，并重写 `load_annotations(self, ann_file)` 以及 `get_ann_info(self, idx)` 这两个方法，正如[CocoDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py)与[VOCDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/voc.py).

- 离线转换（offline conversion）

  可以将标注格式转换为上述的任意格式并将其保存为 pickle 或者 json 文件，例如[pascal_voc.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/pascal_voc.py)。
  然后使用`CustomDataset`。

### 自定义数据集的例子：

假设文本文件中表示的是一种全新的标注格式。边界框的标注信息保存在 `annotation.txt` 中，内容如下：

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

我们可以在 `mmdet/datasets/my_dataset.py` 中创建一个新的 dataset 用以加载数据。

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

配置文件中，可以使用 `MyDataset` 进行如下修改

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

## 使用 dataset 包装器自定义数据集

MMDetection 也支持非常多的数据集包装器（wrapper）来混合数据集或在训练时修改数据集的分布。
最近 MMDetection 支持如下三种数据集包装：

- `RepeatDataset`：将整个数据集简单地重复。
- `ClassBalancedDataset`：以类别均衡的方式重复数据集。
- `ConcatDataset`：合并数据集。

### 重复数据集（Repeat dataset）

使用 `RepeatDataset` 包装器来重复数据集。例如，假设原始数据集为 `Dataset_A`，重复它过后，其配置如下：


```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # Dataset_A 的原始配置信息
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### 类别均衡数据集（Class balanced dataset）

使用 `ClassBalancedDataset` 作为包装器在类别的出现的频率上重复数据集。数据集需要实例化 `self.get_cat_ids(idx)` 函数以支持 `ClassBalancedDataset`。
比如，以 `oversample_thr=1e-3` 来重复数据集 `Dataset_A`，其配置如下：

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # Dataset_A 的原始配置信息
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

更多细节请参考[源码](../../mmdet/datasets/dataset_wrappers.py)。

### 合并数据集（Concatenate dataset）

合并数据集有三种方法：

1. 如果要合并的数据集类型一致但有多个的标注文件，那么可以使用如下配置将其合并。

    ```python
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        pipeline=train_pipeline
    )
    ```

    如果合并的数据集适用于测试或者评估，那么这种方式支持每个数据集分开进行评估。如果想要将合并的数据集作为整体用于评估，那么可以像如下一样设置 `separate_eval=False`。

    ```python
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        separate_eval=False,
        pipeline=train_pipeline
    )
    ```

2. 如果想要合并的是不同数据集，那么可以使用如下配置。

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

    只需设置 `separate_eval=False`，用户就可以将所有的数据集作为一个整体来评估。

**注意**

1. 在做评估时，`separate_eval=False` 选项是假设数据集使用了 `self.data_infos`。因此COCO数据集不支持此项操作，因为COCO数据集在做评估时并不是所有都依赖 `self.data_infos`。组合不同类型的数据集并将其作为一个整体来评估，这种做法没有得到测试，也不建议这样做。

2. 因为不支持评估 `ClassBalancedDataset` 和 `RepeatDataset`，所以也不支持评估它们的组合。

一个更复杂的例子则是分别将 `Dataset_A` 和 `Dataset_B` 重复N和M次，然后进行如下合并。

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

## 修改数据集的类别

根据现有数据集的类型，我们可以修改它们的类别名称来训练其标注的子集。
例如，如果只想训练当前数据集中的三个类别，那么就可以修改数据集的类别元组。
数据集就会自动屏蔽掉其他类别的真实框。

```python
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

MMDetection V2.0 也支持从文件中读取类别名称，这种方式在实际应用中很常见。
假设存在文件 `classes.txt`，其包含了如下的类别名称。

```
person
bicycle
car
```

用户可以将类别设置成文件路径，数据集就会自动将其加载并转换成一个列表。

```python
classes = 'path/to/classes.txt'
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
```

**注意**

- 在 MMDetection v2.5.0 之前，如果类别为集合时数据集将自动过滤掉不包含 GT 的图片，且没办法通过修改配置将其关闭。这是一种不可取的行为而且会引起混淆，因为当类别不是集合时数据集只有在 `filter_empty_gt=True` 以及 `test_mode=False` 的情况下才会过滤掉不包含 GT 的图片。在 MMDetection v2.5.0 之后，我们将图片的过滤以及类别的修改进行解耦，如，数据集只有在 `filter_empty_gt=True` 和 `test_mode=False` 的情况下才会过滤掉不包含 GT 的图片，无论类别是否为集合。设置类别只会影响用于训练的标注类别，用户可以自行决定是否过滤不包含 GT 的图片。
- 因为中间格式只有框的标签并不包含类别的名字，所以使用 `CustomDataset` 时用户不能通过修改配置来过滤不含 GT 的图片。但是可以通过离线的方式来解决。
- 当设置数据集中的 `classes` 时，记得修改 `num_classes`。从 v2.9.0 (PR#4508) 之后，我们实现了[NumClassCheckHook](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/utils.py)来检查类别数是否一致。
- 我们在未来将会重构设置数据集类别以及数据集过滤的特性，使其更加地方便用户使用。

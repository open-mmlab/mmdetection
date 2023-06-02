# 自定义数据集

## 支持新的数据格式

为了支持新的数据格式，可以选择将数据转换成现成的格式（COCO 或者 PASCAL）或将其转换成中间格式。当然也可以选择以离线的形式（在训练之前使用脚本转换）或者在线的形式（实现一个新的 dataset 在训练中进行转换）来转换数据。

在 MMDetection 中，建议将数据转换成 COCO 格式并以离线的方式进行，因此在完成数据转换后只需修改配置文件中的标注数据的路径和类别即可。

### 将新的数据格式转换为现有的数据格式

最简单的方法就是将你的数据集转换成现有的数据格式（COCO 或者 PASCAL VOC）

COCO 格式的 JSON 标注文件有如下必要的字段：

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
            249.06]],  # 如果有 mask 标签且为多边形 XY 点坐标格式，则需要保证至少包括 3 个点坐标，否则为无效多边形
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

在 JSON 文件中有三个必要的键：

- `images`: 包含多个图片以及它们的信息的数组，例如 `file_name`、`height`、`width` 和 `id`。
- `annotations`: 包含多个实例标注信息的数组。
- `categories`: 包含多个类别名字和 ID 的数组。

在数据预处理之后，使用现有的数据格式来训练自定义的新数据集有如下两步（以 COCO 为例）：

1. 为自定义数据集修改配置文件。
2. 检查自定义数据集的标注。

这里我们举一个例子来展示上面的两个步骤，这个例子使用包括 5 个类别的 COCO 格式的数据集来训练一个现有的 Cascade Mask R-CNN R50-FPN 检测器

#### 1. 为自定义数据集修改配置文件

配置文件的修改涉及两个方面：

1. `dataloaer` 部分。需要在 `train_dataloader.dataset`、`val_dataloader.dataset` 和 `test_dataloader.dataset` 中添加 `metainfo=dict(classes=classes)`, 其中 classes 必须是 tuple 类型。
2. `model` 部分中的 `num_classes`。需要将默认值（COCO 数据集中为 80）修改为自定义数据集中的类别数。

`configs/my_custom_config.py` 内容如下：

```python

# 新的配置来自基础的配置以更好地说明需要修改的地方
_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 1. 数据集设定
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
data_root='path/to/your/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # 将类别名字添加至 `metainfo` 字段中
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
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotation_data',
        data_prefix=dict(img='val/image_data')
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # 将类别名字添加至 `metainfo` 字段中
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='test/image_data')
        )
    )

# 2. 模型设置

# 将所有的 `num_classes` 默认值修改为 5（原来为80）
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
            249.06]],  # 如果有 mask 标签。
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

我们使用这种方式来支持 CityScapes 数据集。脚本在 [cityscapes.py](https://github.com/open-mmlab/mmdetection/blob/main/tools/dataset_converters/cityscapes.py) 并且我们提供了微调的 [configs](https://github.com/open-mmlab/mmdetection/blob/main/configs/cityscapes).

**注意**

1. 对于实例分割数据集, **MMDetection 目前只支持评估 COCO 格式的 mask AP**.
2. 推荐训练之前进行离线转换，这样就可以继续使用 `CocoDataset` 且只需修改标注文件的路径以及训练的种类。

### 调整新的数据格式为中间格式

如果不想将标注格式转换为 COCO 或者 PASCAL 格式也是可行的。实际上，我们在 MMEngine 的 [BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L116) 中定义了一种简单的标注格式并且与所有现有的数据格式兼容，也能进行离线或者在线转换。

数据集的标注必须为 `json` 或 `yaml`，`yml` 或 `pickle`，`pkl` 格式；标注文件中存储的字典必须包含 `metainfo` 和 `data_list` 两个字段。其中 `metainfo` 是一个字典，里面包含数据集的元信息，例如类别信息；`data_list` 是一个列表，列表中每个元素是一个字典，该字典定义了一个原始数据（raw data），每个原始数据包含一个或若干个训练/测试样本。

以下是一个 JSON 标注文件的例子:

```json
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
                    "ignore_flag": 1
                  }
                ]
              },
            ...
        ]
}
```

有些数据集可能会提供如：crowd/difficult/ignored bboxes 标注，那么我们使用 `ignore_flag`来包含它们。

在得到上述标准的数据标注格式后，可以直接在配置中使用 MMDetection 的 [BaseDetDataset](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/base_det_dataset.py#L13) ，而无需进行转换。

### 自定义数据集例子

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

配置文件中，可以使用 `MyDataset` 进行如下修改

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

## 使用 dataset 包装器自定义数据集

MMEngine 也支持非常多的数据集包装器（wrapper）来混合数据集或在训练时修改数据集的分布，其支持如下三种数据集包装：

- `RepeatDataset`：将整个数据集简单地重复。
- `ClassBalancedDataset`：以类别均衡的方式重复数据集。
- `ConcatDataset`：合并数据集。

具体使用方式见 [MMEngine 数据集包装器](#TODO)。

## 修改数据集的类别

根据现有数据集的类型，我们可以修改它们的类别名称来训练其标注的子集。
例如，如果只想训练当前数据集中的三个类别，那么就可以修改数据集的 `metainfo` 字典，数据集就会自动屏蔽掉其他类别的真实框。

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

**注意**

- 在 MMDetection v2.5.0 之前，如果类别为集合时数据集将自动过滤掉不包含 GT 的图片，且没办法通过修改配置将其关闭。这是一种不可取的行为而且会引起混淆，因为当类别不是集合时数据集时，只有在 `filter_empty_gt=True` 以及 `test_mode=False` 的情况下才会过滤掉不包含 GT 的图片。在 MMDetection v2.5.0 之后，我们将图片的过滤以及类别的修改进行解耦，数据集只有在 `filter_cfg=dict(filter_empty_gt=True)` 和 `test_mode=False` 的情况下才会过滤掉不包含 GT 的图片，无论类别是否为集合。设置类别只会影响用于训练的标注类别，用户可以自行决定是否过滤不包含 GT 的图片。
- 直接使用 MMEngine 中的 `BaseDataset` 或者 MMDetection 中的 `BaseDetDataset` 时用户不能通过修改配置来过滤不含 GT 的图片，但是可以通过离线的方式来解决。
- 当设置数据集中的 `classes` 时，记得修改 `num_classes`。从 v2.9.0 (PR#4508) 之后，我们实现了 [NumClassCheckHook](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/engine/hooks/num_class_check_hook.py) 来检查类别数是否一致。

## COCO 全景分割数据集

现在我们也支持 COCO Panoptic Dataset，全景注释的格式与 COCO 格式不同，其前景和背景都将存在于注释文件中。COCO Panoptic 格式的注释 JSON 文件具有以下必要的键：

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

此外，`seg` 必须设置为全景注释图像的路径。

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

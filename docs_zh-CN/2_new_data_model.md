# 2: 在自定义数据集上进行训练

通过本文档，你将会知道如何使用自定义的数据集对预定义的模型进行推理，测试以及训练。我们使用[balloon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) 作为例子来描述整个过程。

如下是基本的步骤：

1. 准备自定义数据集
2. 准备配置文件
3. 在自定义数据集上进行训练，测试和推理。

## 准备自定义数据集
MMDetection一共支持三种形式创建新数据集：
1. 将数据集重新组织为COCO格式。
2. 将数据集重新组织为一个中间格式。
3. 实现一个新的数据集。

我们通常建议使用前面两种方法，因为它们通常来说比第三种要简单。
在本文档中，我们展示一个例子来说明将数据转化为COCO格式。
**注意**：MMDetection现只支持对COCO格式的数据集进行mask AP的评测。
所以对于实例分割任务，必须要将数据集转化为COCO格式。

### COCO标注格式
用于实例分割的COCO数据集格式如下所示，其中的键都是必要的，参考[这里](https://cocodataset.org/#format-data)来获取更多细节。
```json
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```

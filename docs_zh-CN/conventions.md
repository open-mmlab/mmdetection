# 默认约定

如果你想把 MMDetection 修改为自己的项目，请遵循下面的约定。

## 损失

在 MMDetection 中，`model(**data)` 的返回值是一个字典，包含着所有的损失和评价指标，他们将会由 `model(**data)` 返回。

例如，在 bbox head 中，

```python
class BBoxHead(nn.Module):
    ...
    def loss(self, ...):
        losses = dict()
        # 分类损失
        losses['loss_cls'] = self.loss_cls(...)
        # 分类准确率
        losses['acc'] = accuracy(...)
        # 边界框损失
        losses['loss_bbox'] = self.loss_bbox(...)
        return losses
```

`'bbox_head.loss()'` 在模型 forward 阶段会被调用。返回的字典中包含了 `'loss_bbox'`,`'loss_cls'`,`'acc'`。只有 `'loss_bbox'`, `'loss_cls'` 会被用于反向传播，`'acc'` 只会被作为评价指标来监控训练过程。

我们默认，只有那些键的名称中包含 `'loss'` 的值会被用于反向传播。这个行为可以通过修改 `BaseDetector.train_step()` 来改变。

## 空 proposals

在 MMDetection 中，我们为两阶段方法中空 proposals 的情况增加了特殊处理和单元测试。我们同时需要处理整个 batch 和单一图片中空 proposals 的情况。例如，在 CascadeRoIHead 中，

```python
# 简单的测试
...

# 在整个 batch中 都没有 proposals
if rois.shape[0] == 0:
    bbox_results = [[
        np.zeros((0, 5), dtype=np.float32)
        for _ in range(self.bbox_head[-1].num_classes)
    ]] * num_imgs
    if self.with_mask:
        mask_classes = self.mask_head[-1].num_classes
        segm_results = [[[] for _ in range(mask_classes)]
                        for _ in range(num_imgs)]
        results = list(zip(bbox_results, segm_results))
    else:
        results = bbox_results
    return results
...

# 在单张图片中没有 proposals
for i in range(self.num_stages):
    ...
    if i < self.num_stages - 1:
          for j in range(num_imgs):
                   # 处理空 proposals
                   if rois[j].shape[0] > 0:
                       bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                       refine_roi = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label[j], bbox_pred[j], img_metas[j])
                       refine_roi_list.append(refine_roi)
```
如果你有自定义的 `RoIHead`, 你可以参考上面的方法来处理空 proposals 的情况。

## 全景分割数据集

在 MMDetection 中，我们支持了 COCO 全景分割数据集 `CocoPanopticDataset`。对于它的实现，我们在这里声明一些默认约定。

1. 在 mmdet<=2.16.0 时，语义分割标注中的前景和背景标签范围与 MMDetection 中的默认规定有所不同。标签 `0` 代表 `VOID` 标签。
从 mmdet=2.17.0 开始，为了和框的类别标注保持一致，语义分割标注的类别标签也改为从 `0` 开始，标签 `255` 代表 `VOID` 类。
为了达成这一目标，我们在流程 `Pad` 里支持了设置 `seg` 的填充值的功能。
2. 在评估中，全景分割结果必须是一个与原图大小相同的图。结果图中每个像素的值有如此形式：`instance_id * INSTANCE_OFFSET + category_id`。

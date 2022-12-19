# Conventions

Please check the following conventions if you would like to modify MMDetection as your own project.

## About the order of image shape

In OpenMMLab 2.0, to be consistent with the input argument of OpenCV, the argument about image shape in the data transformation pipeline is always in the `(width, height)` order. On the contrary, for computation convenience, the order of the field going through the data pipeline and the model is `(height, width)`. Specifically, in the results processed by each data transform pipeline, the fields and their value meaning is as below:

- img_shape: (height, width)
- ori_shape: (height, width)
- pad_shape: (height, width)
- batch_input_shape: (height, width)

As an example, the initialization arguments of `Mosaic` are as below:

```python
@TRANSFORMS.register_module()
class Mosaic(BaseTransform):
    def __init__(self,
                img_scale: Tuple[int, int] = (640, 640),
                center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                bbox_clip_border: bool = True,
                pad_val: float = 114.0,
                prob: float = 1.0) -> None:
       ...

       # img_scale order should be (width, height)
       self.img_scale = img_scale

    def transform(self, results: dict) -> dict:
        ...

        results['img'] = mosaic_img
        # (height, width)
        results['img_shape'] = mosaic_img.shape[:2]
```

## Loss

In MMDetection, a `dict` containing losses and metrics will be returned by `model(**data)`.

For example, in bbox head,

```python
class BBoxHead(nn.Module):
    ...
    def loss(self, ...):
        losses = dict()
        # classification loss
        losses['loss_cls'] = self.loss_cls(...)
        # classification accuracy
        losses['acc'] = accuracy(...)
        # bbox regression loss
        losses['loss_bbox'] = self.loss_bbox(...)
        return losses
```

`bbox_head.loss()` will be called during model forward.
The returned dict contains `'loss_bbox'`, `'loss_cls'`, `'acc'` .
Only `'loss_bbox'`, `'loss_cls'` will be used during back propagation,
`'acc'` will only be used as a metric to monitor training process.

By default, only values whose keys contain `'loss'` will be back propagated.
This behavior could be changed by modifying `BaseDetector.train_step()`.

## Empty Proposals

In MMDetection, We have added special handling and unit test for empty proposals of two-stage. We need to deal with the empty proposals of the entire batch and single image at the same time. For example, in CascadeRoIHead,

```python
# simple_test method
...
# There is no proposal in the whole batch
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

# There is no proposal in the single image
for i in range(self.num_stages):
    ...
    if i < self.num_stages - 1:
          for j in range(num_imgs):
                # Handle empty proposal
                if rois[j].shape[0] > 0:
                    bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                    refine_roi = self.bbox_head[i].regress_by_class(
                         rois[j], bbox_label, bbox_pred[j], img_metas[j])
                    refine_roi_list.append(refine_roi)
```

If you have customized `RoIHead`, you can refer to the above method to deal with empty proposals.

## Coco Panoptic Dataset

In MMDetection, we have supported COCO Panoptic dataset. We clarify a few conventions about the implementation of `CocoPanopticDataset` here.

1. For mmdet\<=2.16.0, the range of foreground and background labels in semantic segmentation are different from the default setting of MMDetection. The label `0` stands for `VOID` label and the category labels start from `1`.
   Since mmdet=2.17.0, the category labels of semantic segmentation start from `0` and label `255` stands for `VOID` for consistency with labels of bounding boxes.
   To achieve that, the `Pad` pipeline supports setting the padding value for `seg`.
2. In the evaluation, the panoptic result is a map with the same shape as the original image. Each value in the result map has the format of `instance_id * INSTANCE_OFFSET + category_id`.

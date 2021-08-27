# Conventions

Please check the following conventions if you would like to modify MMDetection as your own project.

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

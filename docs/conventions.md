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

# Conventions

Please check the following conventions if you would like to modify MMDetection as your own project.

## Loss
In MMDetection, a `dict` containing losses and metrics will be returned by `model(**data)`.
By default, only values whose keys contain `'loss'` will be back propagated.
This behavior could be changed by modifying ``BaseDetector.train_step()``.

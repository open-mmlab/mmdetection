import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.wrappers import NewEmptyTensorOp, obsolete_torch_version

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def adaptive_avg_pool2d(input, output_size):
    """Handle empty batch dimension to adaptive_avg_pool2d.

    Args:
        input (tensor): 4D tensor.
        output_size (int, tuple[int,int]): the target output size.
    """
    if input.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        output_size = [*input.shape[:2], *output_size]
        empty = NewEmptyTensorOp.apply(input, output_size)
        return empty
    else:
        return F.adaptive_avg_pool2d(input, output_size)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Handle empty batch dimension to AdaptiveAvgPool2d."""

    def forward(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            output_size = self.output_size
            if isinstance(output_size, int):
                output_size = [output_size, output_size]
            else:
                output_size = [
                    v if v is not None else d
                    for v, d in zip(output_size,
                                    x.size()[-2:])
                ]
            output_size = [*x.shape[:2], *output_size]
            empty = NewEmptyTensorOp.apply(x, output_size)
            return empty

        return super().forward(x)

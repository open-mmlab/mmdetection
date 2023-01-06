from mmengine.registry import MODELS
from torch import Tensor
from torch.nn import LayerNorm


@MODELS.register_module()
class CustomLN(LayerNorm):
    """Custom Layer Normalization."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = super().forward(x)
        return x

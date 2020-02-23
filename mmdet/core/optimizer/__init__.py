from torch.optim import *  # noqa: F401, F403

from .copy_of_sgd import CopyOfSGD
from .registry import OPTIMIZERS, TORCH_OPTIMIZERS

__all__ = ['OPTIMIZERS', 'CopyOfSGD', *TORCH_OPTIMIZERS]

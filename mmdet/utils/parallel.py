# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel import DataParallel


# TODO: This is a transition plan and will be deleted in the future
class MMDistributedDataParallel(DistributedDataParallel):
    def train_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


class MMDataParallel(DataParallel):
    def train_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)


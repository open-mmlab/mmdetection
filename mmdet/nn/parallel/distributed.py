from torch.nn.parallel import DistributedDataParallel

from .scatter_gather import scatter_kwargs


class MMDistributedDataParallel(DistributedDataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

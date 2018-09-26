from torch.nn.parallel import DataParallel

from .scatter_gather import scatter_kwargs


class MMDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

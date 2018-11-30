import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but
    concat the group flag for image aspect ratio.
    """
    def __init__(self, datasets):
        """
        flag: Images with aspect ratio greater than 1 will be set as group 1,
              otherwise group 0.
        """
        super(ConcatDataset, self).__init__(datasets)
        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)

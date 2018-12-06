import numpy as np


class RepeatDataset(object):

    def __init__(self, dataset, repeat_times):
        self.dataset = dataset
        self.repeat_times = repeat_times
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, repeat_times)
        self.length = len(self.dataset) * self.repeat_times

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return self.length


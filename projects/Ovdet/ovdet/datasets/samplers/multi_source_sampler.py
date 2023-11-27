# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.datasets import GroupMultiSourceSampler
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class CustomGroupMultiSourceSampler(GroupMultiSourceSampler):

    def _get_source_group_info(self) -> None:
        num_sources = len(self.num_per_source)
        self.group2size_per_source = [{0: 0, 1: 0} for _ in range(num_sources)]
        self.group2inds_per_source = [{
            0: [],
            1: []
        } for _ in range(num_sources)]
        for source, dataset in enumerate(self.dataset.datasets):
            for idx in range(len(dataset)):
                data_info = dataset.get_data_info(idx)
                width, height = data_info['width'], data_info['height']
                group = 0 if width < height else 1
                self.group2size_per_source[source][group] += 1
                self.group2inds_per_source[source][group].append(idx)

        self.group_sizes = np.zeros(2, dtype=np.int64)
        for group2size in self.group2size_per_source:
            for group, size in group2size.items():
                self.group_sizes[group] += size
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from mmengine.dataset import Compose
from torch.utils.data import Dataset as TorchDataset

from mmdet.registry import DATASETS


def _decode_batched_data(batched_data):
    data_list = []
    for values in zip(*batched_data.values()):
        data_list.append(dict(zip(batched_data.keys(), values)))
    return data_list


def _encode_batched_data(data_list):
    batched_data = {}
    for key in data_list[0].keys():
        batched_data[key] = [data[key] for data in data_list]
    return batched_data


@DATASETS.register_module()
class HuggingFaceDataset(TorchDataset):

    def __init__(self,
                 repo: str,
                 split: str = 'train',
                 pipeline: Optional[List[Union[dict, Callable]]] = None,
                 classes: Optional[Union[List[str], str]] = None,
                 streaming: bool = False,
                 max_samples: Optional[int] = None,
                 *args,
                 lazy_init: bool = True,
                 max_refetch: int = 20,
                 **kwargs):
        if pipeline is None:
            pipeline = []

        self.repo = repo
        self.pipeline = Compose(pipeline)
        self.streaming = streaming
        self.max_samples = max_samples

        self._lazy_init = lazy_init
        self.max_refetch = max_refetch

        self.dataset: Union[HFDataset, HFIterableDataset] = load_dataset(
            self.repo,
            *args,
            split=split,
            **kwargs,
        )

        self.metainfo = {}

        if isinstance(classes, List):
            self.metainfo['classes'] = classes
        if isinstance(classes, str):
            self.metainfo['classes'] = self._get_feature(classes).names
        else:
            self.metainfo['classes'] = self._get_feature(
                'objects.category').names

        if self._lazy_init:
            self.dataset.set_transform(self.prepare_batched_data)
        else:
            self.dataset.map(self.prepare_data)

        super().__init__()

    def _get_feature(self, key):
        if isinstance(key, str):
            key = key.split('.')
        feature = self.dataset
        for k in key[:-1]:
            feature = feature.features[k]
        return feature.feature[key[-1]]

    def prepare_batched_data(self, batched_data):
        data_list = _decode_batched_data(batched_data)
        data_list = [self.prepare_data(data) for data in data_list]
        return _encode_batched_data(data_list)

    def prepare_data(self, item):
        result = self.pipeline(item)

        # XXX: There’s a randomly occurring bug here.
        #      after going through self.pipeline,
        #      there’s a significant chance it returns None.
        #      The same content can be fixed by calling self.pipeline again.
        if result is None:
            for _ in range(self.max_refetch):
                result = self.pipeline(item)
                if result is not None:
                    break
            else:
                err = f'returned None from dataset at item {item}.'
                raise RuntimeError(err)

        return result

    def __getitem__(self, idx):
        if self.max_samples is not None and idx >= self.max_samples:
            raise IndexError(
                f'Index {idx} out of range for dataset of length {len(self)}.')

        return self.dataset[idx]

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            yield self.dataset[idx]

    def __len__(self):
        if isinstance(self.dataset, HFIterableDataset):
            if self.max_samples is None:
                raise ValueError('Cannot get length of streaming dataset. '
                                 'Please specify `max_samples`.')
            return self.max_samples
        else:
            if self.max_samples is not None:
                return min(len(self.dataset), self.max_samples)
            return len(self.dataset)

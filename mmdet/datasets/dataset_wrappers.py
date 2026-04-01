# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import List, Sequence, Union

from mmengine.dataset import BaseDataset
from mmengine.dataset import ConcatDataset as MMENGINE_ConcatDataset
from mmengine.dataset import force_full_init

from mmdet.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None. It is deprecated.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Default: 15.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 pipeline: Sequence[str],
                 skip_type_keys: Union[Sequence[str], None] = None,
                 max_refetch: int = 15,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')

        self._metainfo = self.dataset.metainfo
        if hasattr(self.dataset, 'flag'):
            self.flag = self.dataset.flag
        self.num_samples = len(self.dataset)
        self.max_refetch = max_refetch

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self.dataset)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')

            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys. It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys


@DATASETS.register_module()
class ConcatDataset(MMENGINE_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset``, support
    lazy_init and get_dataset_source.

    Note:
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # if the metainfo of multiple datasets are the same, use metainfo
        # of the first dataset, else try to merge the class lists from
        # different datasets into a unified metainfo dict
        is_all_same = True
        self._metainfo_first = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    is_all_same = False
                    break
                if self._metainfo_first[key] != dataset.metainfo[key]:
                    is_all_same = False
                    break

        if is_all_same:
            self._metainfo = self.datasets[0].metainfo
        else:
            self._metainfo = self._merge_metainfo()

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

            if isinstance(self._metainfo, dict):
                self._metainfo.update(
                    dict(cumulative_sizes=self.cumulative_sizes))
            else:
                for i, dataset in enumerate(self.datasets):
                    self._metainfo[i].update(
                        dict(cumulative_sizes=self.cumulative_sizes))

    def _merge_metainfo(self) -> Union[dict, List[dict]]:
        """Merge metainfo from all sub-datasets.

        When datasets have different ``classes`` (e.g. concatenating a
        CocoDataset with a VOCDataset), merge the class lists into a
        unified set so that downstream evaluation receives a single
        consistent metainfo dict instead of a list.

        Returns:
            Union[dict, List[dict]]: Merged metainfo dict when class
                merging succeeds, otherwise a list of per-dataset
                metainfo dicts (original fallback behaviour).
        """
        # Check whether all datasets carry a 'classes' key
        all_have_classes = all(
            'classes' in ds.metainfo for ds in self.datasets)
        if not all_have_classes:
            return [dataset.metainfo for dataset in self.datasets]

        # Build the merged class list preserving insertion order
        merged_classes: List[str] = []
        seen: set = set()
        for dataset in self.datasets:
            for cls_name in dataset.metainfo['classes']:
                if cls_name not in seen:
                    merged_classes.append(cls_name)
                    seen.add(cls_name)

        # Build a merged palette that matches the merged class order.
        # Re-use palette colours from whichever dataset already defines
        # a colour for a given class; generate a deterministic fallback
        # colour for any class that has no palette entry.
        cls_to_palette: dict = {}
        for dataset in self.datasets:
            ds_classes = dataset.metainfo.get('classes', ())
            ds_palette = dataset.metainfo.get('palette', None)
            if ds_palette is not None and len(ds_palette) == len(ds_classes):
                for cls_name, colour in zip(ds_classes, ds_palette):
                    if cls_name not in cls_to_palette:
                        cls_to_palette[cls_name] = colour

        merged_palette: List[tuple] = []
        for i, cls_name in enumerate(merged_classes):
            if cls_name in cls_to_palette:
                merged_palette.append(cls_to_palette[cls_name])
            else:
                # deterministic fallback colour derived from class index
                merged_palette.append(
                    ((i * 97 + 23) % 256,
                     (i * 53 + 89) % 256,
                     (i * 131 + 47) % 256))

        # Start from the first dataset's metainfo and override class
        # related fields with the merged versions.
        merged = copy.deepcopy(self.datasets[0].metainfo)
        merged['classes'] = tuple(merged_classes)
        merged['palette'] = merged_palette
        return merged

    def get_dataset_source(self, idx: int) -> int:
        dataset_idx, _ = self._get_ori_dataset_idx(idx)
        return dataset_idx

import copy
import glob
import os

from mmdet.utils import build_from_cfg
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .registry import DATASETS


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)

    if cfg.get('img_prefix_auto', False):
        del cfg['img_prefix_auto']
        assert img_prefixes is None
        if cfg['type'] == 'CustomCocoDataset':
            # assuming following dataset structure:
            # dataset_root
            # ├── annotations
            # │   ├── instances_train.json
            # │   ├── ...
            # ├── images
            #     ├── image_name1
            #     ├── image_name2
            #     ├── ...
            # and file_name inside instances_train.json is relative to dataset root
            img_prefixes = [os.path.join(os.path.dirname(ann_file), '..') for ann_file in ann_files]
        else:
            raise NotImplementedError

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg['ann_file'], (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif '*' in cfg['ann_file']:
        cfg['ann_file'] = glob.glob(cfg['ann_file'], recursive=True)
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

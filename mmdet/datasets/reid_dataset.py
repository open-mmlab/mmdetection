# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.utils import check_file_exist

from mmdet.registry import DATASETS


@DATASETS.register_module()
class ReIDDataset(BaseDataset):
    """Dataset for ReID.

    Args:
        triplet_sampler (dict, optional): The sampler for hard mining
            triplet loss. Defaults to None.
        keys: num_ids (int): The number of person ids.
              ins_per_id (int): The number of image for each person.
    """

    def __init__(self, triplet_sampler: dict = None, *args, **kwargs):
        self.triplet_sampler = triplet_sampler
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ''self.ann_file''.

        Returns:
              list[dict]: A list of annotation.
        """
        assert isinstance(self.ann_file, str)
        check_file_exist(self.ann_file)
        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = dict(img_prefix=self.data_prefix)
                if self.data_prefix['img_path'] is not None:
                    info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                filename)
                else:
                    info['img_path'] = filename
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_list.append(info)
        self._parse_ann_info(data_list)
        return data_list

    def _parse_ann_info(self, data_list: List[dict]):
        """Parse person id annotations."""
        index_tmp_dic = defaultdict(list)  # pid->[idx1,...,idxN]
        self.index_dic = dict()  # pid->array([idx1,...,idxN])
        for idx, info in enumerate(data_list):
            pid = info['gt_label']
            index_tmp_dic[int(pid)].append(idx)
        for pid, idxs in index_tmp_dic.items():
            self.index_dic[pid] = np.asarray(idxs, dtype=np.int64)
        self.pids = np.asarray(list(self.index_dic.keys()), dtype=np.int64)

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ''self.pipeline''.

        Args:
            idx (int): The index of ''data_info''

        Returns:
            Any: Depends on ''self.pipeline''
        """
        data_info = self.get_data_info(idx)
        if self.triplet_sampler is not None:
            img_info = self.triplet_sampling(data_info['gt_label'],
                                             **self.triplet_sampler)
            data_info = copy.deepcopy(img_info)  # triplet -> list
        else:
            data_info = copy.deepcopy(data_info)  # no triplet -> dict
        return self.pipeline(data_info)

    def triplet_sampling(self,
                         pos_pid,
                         num_ids: int = 8,
                         ins_per_id: int = 4) -> Dict:
        """Triplet sampler for hard mining triplet loss. First, for one
        pos_pid, random sample ins_per_id images with same person id.

        Then, random sample num_ids - 1 images for each negative id.
        Finally, random sample ins_per_id images for each negative id.

        Args:
            pos_pid (ndarray): The person id of the anchor.
            num_ids (int): The number of person ids.
            ins_per_id (int): The number of images for each person.

        Returns:
            Dict: Annotation information of num_ids X ins_per_id images.
        """
        assert len(self.pids) >= num_ids, \
            'The number of person ids in the training set must ' \
            'be greater than the number of person ids in the sample.'

        pos_idxs = self.index_dic[int(
            pos_pid)]  # all positive idxs for pos_pid
        idxs_list = []
        # select positive samplers
        idxs_list.extend(pos_idxs[np.random.choice(
            pos_idxs.shape[0], ins_per_id, replace=True)])
        # select negative ids
        neg_pids = np.random.choice(
            [i for i, _ in enumerate(self.pids) if i != pos_pid],
            num_ids - 1,
            replace=False)
        # select negative samplers for each negative id
        for neg_pid in neg_pids:
            neg_idxs = self.index_dic[neg_pid]
            idxs_list.extend(neg_idxs[np.random.choice(
                neg_idxs.shape[0], ins_per_id, replace=True)])
        # return the final triplet batch
        triplet_img_infos = []
        for idx in idxs_list:
            triplet_img_infos.append(copy.deepcopy(self.get_data_info(idx)))
        # Collect data_list scatters (list of dict -> dict of list)
        out = dict()
        for key in triplet_img_infos[0].keys():
            out[key] = [_info[key] for _info in triplet_img_infos]
        return out

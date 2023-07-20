# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs

from ..registry import DATASETS


@DATASETS.register_module()
class BaseDetDataset(BaseDataset):
    """Base dataset for detection.

    Args:
        proposal_file (str, optional): Proposals file path. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
    """

    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)

    def full_init(self) -> None:
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - load_proposals: Load proposals from proposal file, if
              `self.proposal_file` is not None.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # get proposals from file
        if self.proposal_file is not None:
            self.load_proposals()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def load_proposals(self) -> None:
        """Load proposals from proposals file.

        The `proposals_list` should be a dict[img_path: proposals]
        with the same length as `data_list`. And the `proposals` should be
        a `dict` or :obj:`InstanceData` usually contains following keys.

            - bboxes (np.ndarry): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
            - scores (np.ndarry): Classification scores, has a shape
              (num_instance, ).
        """
        # TODO: Add Unit Test after fully support Dump-Proposal Metric
        if not is_abs(self.proposal_file):
            self.proposal_file = osp.join(self.data_root, self.proposal_file)
        proposals_list = load(
            self.proposal_file, backend_args=self.backend_args)
        assert len(self.data_list) == len(proposals_list)
        for data_info in self.data_list:
            img_path = data_info['img_path']
            # `file_name` is the key to obtain the proposals from the
            # `proposals_list`.
            file_name = osp.join(
                osp.split(osp.split(img_path)[0])[-1],
                osp.split(img_path)[-1])
            proposals = proposals_list[file_name]
            data_info['proposals'] = proposals

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

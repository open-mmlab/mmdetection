# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os.path as osp
import warnings
from typing import List, Union

import mmcv
from mmengine.dist import get_rank
from mmengine.fileio import dump, get, get_text, load
from mmengine.logging import print_log
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class CrowdHumanDataset(BaseDetDataset):
    r"""Dataset for CrowdHuman.

    Args:
        data_root (str): The root directory for
            ``data_prefix`` and ``ann_file``.
        ann_file (str): Annotation file path.
        extra_ann_file (str | optional):The path of extra image metas
            for CrowdHuman. It can be created by CrowdHumanDataset
            automatically or by tools/misc/get_crowdhuman_id_hw.py
            manually. Defaults to None.
    """

    METAINFO = {
        'classes': ('person', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60)]
    }

    def __init__(self, data_root, ann_file, extra_ann_file=None, **kwargs):
        # extra_ann_file record the size of each image. This file is
        # automatically created when you first load the CrowdHuman
        # dataset by mmdet.
        if extra_ann_file is not None:
            self.extra_ann_exist = True
            self.extra_anns = load(extra_ann_file)
        else:
            ann_file_name = osp.basename(ann_file)
            if 'train' in ann_file_name:
                self.extra_ann_file = osp.join(data_root, 'id_hw_train.json')
            elif 'val' in ann_file_name:
                self.extra_ann_file = osp.join(data_root, 'id_hw_val.json')
            self.extra_ann_exist = False
            if not osp.isfile(self.extra_ann_file):
                print_log(
                    'extra_ann_file does not exist, prepare to collect '
                    'image height and width...',
                    level=logging.INFO)
                self.extra_anns = {}
            else:
                self.extra_ann_exist = True
                self.extra_anns = load(self.extra_ann_file)
        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        anno_strs = get_text(
            self.ann_file, backend_args=self.backend_args).strip().split('\n')
        print_log('loading CrowdHuman annotation...', level=logging.INFO)
        data_list = []
        prog_bar = ProgressBar(len(anno_strs))
        for i, anno_str in enumerate(anno_strs):
            anno_dict = json.loads(anno_str)
            parsed_data_info = self.parse_data_info(anno_dict)
            data_list.append(parsed_data_info)
            prog_bar.update()
        if not self.extra_ann_exist and get_rank() == 0:
            #  TODO: support file client
            try:
                dump(self.extra_anns, self.extra_ann_file, file_format='json')
            except:  # noqa
                warnings.warn(
                    'Cache files can not be saved automatically! To speed up'
                    'loading the dataset, please manually generate the cache'
                    ' file by file tools/misc/get_crowdhuman_id_hw.py')

            print_log(
                f'\nsave extra_ann_file in {self.data_root}',
                level=logging.INFO)

        del self.extra_anns
        print_log('\nDone', level=logging.INFO)
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.data_prefix['img'],
                            f"{raw_data_info['ID']}.jpg")
        data_info['img_path'] = img_path
        data_info['img_id'] = raw_data_info['ID']

        if not self.extra_ann_exist:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            data_info['height'], data_info['width'] = img.shape[:2]
            self.extra_anns[raw_data_info['ID']] = img.shape[:2]
            del img, img_bytes
        else:
            data_info['height'], data_info['width'] = self.extra_anns[
                raw_data_info['ID']]

        instances = []
        for i, ann in enumerate(raw_data_info['gtboxes']):
            instance = {}
            if ann['tag'] not in self.metainfo['classes']:
                instance['bbox_label'] = -1
                instance['ignore_flag'] = 1
            else:
                instance['bbox_label'] = self.metainfo['classes'].index(
                    ann['tag'])
                instance['ignore_flag'] = 0
            if 'extra' in ann:
                if 'ignore' in ann['extra']:
                    if ann['extra']['ignore'] != 0:
                        instance['bbox_label'] = -1
                        instance['ignore_flag'] = 1

            x1, y1, w, h = ann['fbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            instance['bbox'] = bbox

            # Record the full bbox(fbox), head bbox(hbox) and visible
            # bbox(vbox) as additional information. If you need to use
            # this information, you just need to design the pipeline
            # instead of overriding the CrowdHumanDataset.
            instance['fbox'] = bbox
            hbox = ann['hbox']
            instance['hbox'] = [
                hbox[0], hbox[1], hbox[0] + hbox[2], hbox[1] + hbox[3]
            ]
            vbox = ann['vbox']
            instance['vbox'] = [
                vbox[0], vbox[1], vbox[0] + vbox[2], vbox[1] + vbox[3]
            ]

            instances.append(instance)

        data_info['instances'] = instances
        return data_info

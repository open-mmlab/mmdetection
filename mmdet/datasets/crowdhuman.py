# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Union

import mmcv
from mmengine.dataset import BaseDataset
from mmengine.fileio import FileClient

from mmdet.registry import DATASETS


@DATASETS.register_module()
class CrowdHumanDataset(BaseDataset):
    """Dataset for CrowdHuman."""

    METAINFO = {
        'CLASSES': ('person', ),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(220, 20, 60)]
    }

    def __init__(self, file_client_args: dict = dict(backend='disk'),
                 **kwargs):
        assert 'data_root' in kwargs, \
            f"{self.__class__.__name__} needs parameter: 'data_root' \n"
        assert osp.exists(kwargs['data_root']), \
            "'data_root' is not valid: Please enter the correct parameter \n"
        self.id_hw_path = osp.join(kwargs['data_root'], 'id_hw.json')
        self.id_hw_exist_flag = True
        if not osp.isfile(self.id_hw_path):
            self.file_client = FileClient(**file_client_args)
            self.id_hw_exist_flag = False
            self.id_hw = {}
        else:
            with open(self.id_hw_path, 'r') as id_hw_file:
                self.id_hw = json.load(id_hw_file)
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with open(self.ann_file, 'r') as file:
            anno_strs = file.readlines()

        print('loading CrowdHuman annotation...')
        data_list = []
        prog_bar = mmcv.ProgressBar(len(anno_strs))
        for i, anno_str in enumerate(anno_strs):
            anno_dict = json.loads(anno_str)
            parsed_data_info = self.parse_data_info(anno_dict)
            data_list.append(parsed_data_info)
            prog_bar.update()
        if not self.id_hw_exist_flag:
            with open(self.id_hw_path, 'w', encoding='utf-8') as file:
                json.dump(self.id_hw, file, indent=4)
            print(f'\nsave id_hw in {self.data_root}')
        del self.id_hw, self.id_hw_exist_flag, self.id_hw_path
        print('Done')
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

        if not self.id_hw_exist_flag:
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            data_info['height'], data_info['width'] = img.shape[:2]
            self.id_hw[raw_data_info['ID']] = img.shape[:2]
            del img, img_bytes
        else:
            data_info['height'], data_info['width'] = self.id_hw[
                raw_data_info['ID']]

        instances = []
        for i, ann in enumerate(raw_data_info['gtboxes']):
            instance = {}
            if ann['tag'] == 'mask':
                continue
            x1, y1, w, h = ann['fbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = 0

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


if __name__ == '__main__':
    train_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='BoxesCrop'),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackDetInputs')
    ]
    dataset = CrowdHumanDataset(
        data_root=r'D:\BaiduNetdiskDownload\CrowdHuman',
        ann_file='annotation_train.odgt',
        data_prefix=dict(img='Images/'),
        pipeline=train_pipeline)
    data = dataset.__getitem__(1)
    print(data)
    print('down!')

# if __name__ == '__main__':
#     dataset = CrowdHumanDataset(
#         data_root=r'C:\Users\25469\Desktop',
#         ann_file='test_annotation_train.odgt',
#         data_prefix=dict(img='Images/'),
#         # filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=[])
#     print(len(dataset))
#     print('down!')

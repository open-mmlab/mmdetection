# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

from mmengine.dist import is_main_process
from mmengine.fileio import get_local_path, list_from_file
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS
from mmdet.utils.typing_utils import List, Union
from .xml_style import XMLDataset


@DATASETS.register_module()
class WIDERFaceDataset(XMLDataset):
    """Reader for the WIDER Face dataset in PASCAL VOC format.

    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    METAINFO = {'classes': ('face', ), 'palette': [(0, 255, 0)]}

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)

        # loading process takes around 10 mins
        if is_main_process():
            prog_bar = ProgressBar(len(img_ids))

        for img_id in img_ids:
            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = f'{img_id}.jpg'
            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)

            if is_main_process():
                prog_bar.update()
        return data_list

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_id = img_info['img_id']
        xml_path = osp.join(self.data_prefix['img'], 'Annotations',
                            f'{img_id}.xml')
        data_info['img_id'] = img_id
        data_info['xml_path'] = xml_path

        # deal with xml file
        with get_local_path(
                xml_path, backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        folder = root.find('folder').text
        img_path = osp.join(self.data_prefix['img'], folder,
                            img_info['file_name'])
        data_info['img_path'] = img_path

        data_info['height'] = height
        data_info['width'] = width

        # Coordinates are in range [0, width - 1 or height - 1]
        data_info['instances'] = self._parse_instance_info(
            raw_ann_info, minus_one=False)
        return data_info

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

from mmengine.dist import is_main_process
from mmengine.fileio import list_from_file
from mmengine.utils import ProgressBar

from mmdet.registry import DATASETS
from mmdet.utils.typing_utils import List
from .xml_style import XMLDataset


@DATASETS.register_module()
class WIDERFaceDataset(XMLDataset):
    """Reader for the WIDER Face dataset in PASCAL VOC format.

    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    METAINFO = {'classes': ('face', ), 'palette': [(0, 255, 0)]}

    def __init__(self, **kwargs):
        super(WIDERFaceDataset, self).__init__(
            img_subdir='', minus_one=False, **kwargs)

    @property
    def img_root(self) -> str:
        """Return the sub data root."""
        return self.data_prefix.get('img', '')

    def load_data_list(self) -> List[dict]:
        """Load annotation from WIDERFace XML style annotation file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        img_ids = list_from_file(
            self.ann_file, file_client_args=self.file_client_args)

        # loading process takes around 10 mins
        if is_main_process():
            prog_bar = ProgressBar(len(img_ids))

        for img_id in img_ids:
            xml_path = osp.normpath(
                osp.join(self.img_root, self.ann_subdir, f'{img_id}.xml'))
            with self.file_client.get_local_path(xml_path) as local_path:
                raw_ann_info = ET.parse(local_path)
            root = raw_ann_info.getroot()
            folder = root.find('folder').text
            filename = root.find('filename').text
            assert osp.splitext(filename)[0] == img_id
            img_path = osp.normpath(
                osp.join(self.img_root, self.img_subdir, folder, filename))

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['img_path'] = img_path
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)

            if is_main_process():
                prog_bar.update()

        if is_main_process():
            prog_bar.file.write('\n')

        return data_list

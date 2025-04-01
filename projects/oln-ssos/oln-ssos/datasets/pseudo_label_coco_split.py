import copy
import os.path as osp
from typing import Union, List

from mmengine import get_local_path

from mmdet.registry import DATASETS
from projects.OLN.oln import CocoSplitDataset


@DATASETS.register_module()
class PseudoLabelCocoSplitDataset(CocoSplitDataset):

    def __init__(self, *args, **kwargs):
        self.cat_pseudo_label_mapping = None
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        for ann in self.coco.anns.values():
            if self.cat_pseudo_label_mapping is None:
                ann['pseudo_label'] = 0
            else:
                if ann['id'] in self.cat_pseudo_label_mapping.keys():
                    ann['pseudo_label'] = int(self.cat_pseudo_label_mapping[ann['id']])
                else:
                    ann['pseudo_label'] = 0

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.train_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.train_class]
        )
        self.eval_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.eval_class]
        )
        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                    raw_ann_info,
                'raw_img_info':
                    raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        # print(f"Returning data_list: {data_list}")

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.train_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
                instance['gt_bbox_ignore'] = bbox
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            instance['ann_id'] = ann['id']
            instance['pseudo_label'] = ann['pseudo_label']

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
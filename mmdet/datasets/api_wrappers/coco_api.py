# Copyright (c) OpenMMLab. All rights reserved.
# This file add snake case alias for coco api

import warnings
from collections import defaultdict
from typing import List, Optional, Union

import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        cat_ids_coco = self.getCatIds(cat_names, sup_names, cat_ids)
        if None in cat_names:
            index = [i for i, v in enumerate(cat_names) if v is not None]
            cat_ids = list(range(len(cat_names)))
            for i in range(len(index)):
                cat_ids[index[i]] = cat_ids_coco[i]
            return cat_ids
        else:
            return cat_ids_coco

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


# just for the ease of import
COCOeval = _COCOeval


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

    def __init__(self, annotation_file: Optional[str] = None) -> None:
        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self) -> None:
        """Create index."""
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    # annotations from different images but
                    # may have same segment_id
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

            # filter out annotations from other images
            img_to_anns_ = defaultdict(list)
            for k, v in img_to_anns.items():
                img_to_anns_[k] = [x for x in v if x['image_id'] == k]
            img_to_anns = img_to_anns_

        if 'images' in self.dataset:
            for img_info in self.dataset['images']:
                img_info['segm_file'] = img_info['file_name'].replace(
                    'jpg', 'png')
                imgs[img_info['id']] = img_info

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self,
                  ids: Union[List[int], int] = []) -> Optional[List[dict]]:
        """Load anns with the specified ids.

        ``self.anns`` is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (Union[List[int], int]): Integer ids specifying anns.

        Returns:
            anns (List[dict], optional): Loaded ann objects.
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]

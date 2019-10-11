import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .coco import CocoDataset
import cv2
from .registry import DATASETS


@DATASETS.register_module
class CrossroadDataset(CocoDataset):
    CLASSES = ('vehicle', 'person', 'non-vehicle')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        print(self.cat_ids)
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }

        print(self.cat2label)

        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        parse_ann_info = self._parse_ann_info(ann_info)
        #print (parse_ann_info)
        #print(self.img_infos[idx]['image'])
        #image = cv2.imread('data/crossroad' + self.img_infos[idx]['image'][len('/home/yury/pai_cluster_fs/user/yurygoru/crossroad_extra'):])
        #if image is None:
        #    return parse_ann_info
        #for b,l in zip (parse_ann_info['bboxes'], parse_ann_info['labels']):
        #    if l == 0:
        #        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (255,0,0), 2)
        #    elif l ==1 :
        #        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (0,255,0), 2)
        #    elif l ==2 :
        #        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (0,0,255), 2)
        #    else:
        #        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (255,255,255), 2)
        #cv2.imwrite("/tmp/debug/viz/" + str(idx) + '.png', image)
        # cv2.waitKey(0)
        return parse_ann_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue

            if ann['category_id'] == -1:  # exclude background class
                continue

            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann


@DATASETS.register_module
class CrossroadPersonDataset(CrossroadDataset):
    CLASSES = ('person', )

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        # self.cat_ids = self.coco.getCatIds()
        self.cat2label = {0: 0, 2: 1}
        self.cat_ids = [2, ]
        print(self.cat_ids)
        print(self.cat2label)

        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        parse_ann_info = self._parse_ann_info(ann_info)
        # print (parse_ann_info)
        # image = cv2.imread(self.img_infos[idx]['image'])
        # if image is None:
        #     return parse_ann_info
        # for b,l in zip (parse_ann_info['bboxes'], parse_ann_info['labels']):
        #     if l == 0:
        #         cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (255,0,0), 2)
        #     elif l ==1 :
        #         cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (0,255,0), 2)
        #     elif l ==2 :
        #         cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (0,0,255), 2)
        #     else:
        #         cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]),  (255,255,255), 2)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        return parse_ann_info

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        VEHICLE = 1
        PEDESTRIAN = 2
        NONVEHICLE = 3

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue

            if ann['category_id'] in {-1, VEHICLE}:  # exclude background class
                continue

            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd'] or ann['category_id'] == NONVEHICLE:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # print(ann['category_id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

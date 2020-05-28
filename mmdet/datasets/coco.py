import numpy as np
import os
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
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
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

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

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


@DATASETS.register_module
class ConcatenatedCocoDataset(CocoDataset):
    def __init__(self, concatenated_dataset):
        for dataset in concatenated_dataset.datasets:
            assert isinstance(dataset, CocoDataset), type(dataset)
            assert dataset.cat_ids == concatenated_dataset.datasets[0].cat_ids
            assert dataset.cat2label == concatenated_dataset.datasets[0].cat2label
            assert str(dataset.pipeline) == str(concatenated_dataset.datasets[0].pipeline), f'{dataset.pipeline}'
            assert dataset.proposals == concatenated_dataset.datasets[0].proposals

        self.CLASSES = concatenated_dataset.datasets[0].CLASSES
        self.test_mode = concatenated_dataset.datasets[0].test_mode
        self.filter_empty_gt = concatenated_dataset.datasets[0].filter_empty_gt
        self.cat_ids = concatenated_dataset.datasets[0].cat_ids
        self.cat2label = concatenated_dataset.datasets[0].cat2label
        self.pipeline = concatenated_dataset.datasets[0].pipeline
        self.proposals = concatenated_dataset.datasets[0].proposals
        self.img_ids = []
        self.img_infos = []
        self.flag = None
        self.ann_infos = []
        self.img_prefix = None
        self.seg_prefix = None
        self.proposal_file = None
        self.coco = None

        for dataset in concatenated_dataset.datasets:
            img_shift = 0 if not self.img_ids else max(self.img_ids) + 1

            for img_id in dataset.img_ids:
                self.img_ids.append(img_id + img_shift)

            for im_info in dataset.img_infos:
                im_info = im_info
                im_info['id'] += img_shift
                im_info['filename'] = os.path.join(dataset.img_prefix, im_info['filename'])
                self.img_infos.append(im_info)

            if self.coco is None:
                self.coco = dataset.coco
                self.coco.dataset = {'images': dataset.coco.dataset['images'],
                                     'categories': dataset.coco.dataset['categories']}
            else:
                for cat in dataset.coco.catToImgs:
                    self.coco.catToImgs[cat].extend([img_id + img_shift for img_id in dataset.coco.catToImgs[cat]])

                ann_shift = max(self.coco.anns) + 1
                for k, v in dataset.coco.anns.items():
                    v['image_id'] += img_shift
                    v['id'] += ann_shift
                    self.coco.anns[k + ann_shift] = v

                for k, v in dataset.coco.imgs.items():
                    v['id'] += img_shift
                    self.coco.imgs[k + img_shift] = v

                for k, v in dataset.coco.imgToAnns.items():
                    # indices in annotations have been changed above
                    self.coco.imgToAnns[k + img_shift] = v

                for v in dataset.coco.dataset['images']:
                    v['id'] += img_shift
                    self.coco.dataset['images'].append(v)

            if hasattr(dataset, 'flag'):
                if self.flag is None:
                    self.flag = dataset.flag
                else:
                    self.flag = np.concatenate(dataset.flag, axis=0)

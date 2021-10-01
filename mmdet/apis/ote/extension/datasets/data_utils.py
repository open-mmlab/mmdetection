import json
import os.path as osp
from typing import List, Optional

import numpy as np
from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from pycocotools.coco import COCO


def get_classes_from_annotation(path):
    with open(path) as read_file:
        content = json.load(read_file)
        categories = [
            v["name"] for v in sorted(content["categories"], key=lambda x: x["id"])
        ]
    return categories


class LoadAnnotations:
    def __init__(self, with_bbox=True, with_label=True):
        self.with_bbox = with_bbox
        self.with_label = with_label

    def _load_bboxes(self, results):
        ann_info = results["ann_info"]
        results["gt_bboxes"] = ann_info["bboxes"].copy()

        gt_bboxes_ignore = ann_info.get("bboxes_ignore", None)
        if gt_bboxes_ignore is not None:
            results["gt_bboxes_ignore"] = gt_bboxes_ignore.copy()
            results["bbox_fields"].append("gt_bboxes_ignore")
        results["bbox_fields"].append("gt_bboxes")
        return results

    def _load_labels(self, results):
        results["gt_labels"] = results["ann_info"]["labels"].copy()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label})"
        return repr_str


class CocoDataset:
    def __init__(
        self,
        ann_file,
        classes=None,
        data_root=None,
        img_prefix="",
        test_mode=False,
        filter_empty_gt=True,
        min_size=None,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.classes = self.get_classes(classes)
        self.min_size = min_size

        if self.data_root is not None:
            # if not osp.isabs(self.ann_file):
            #     self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        self.data_infos = self.load_annotations(self.ann_file)

        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def __len__(self):
        return len(self.data_infos)

    def pre_pipeline(self, results):
        results["img_prefix"] = self.img_prefix
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        return self.prepare_img(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def prepare_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return LoadAnnotations()(results)

    def get_classes(self, classes=None):
        if classes is None:
            return get_classes_from_annotation(self.ann_file)

        if isinstance(classes, (tuple, list)):
            return classes

        raise ValueError(f"Unsupported type {type(classes)} of classes.")

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.classes)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann["category_id"] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_["image_id"] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if self.min_size is not None:
                if w < self.min_size or h < self.min_size:
                    continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks_ann.append(ann.get("segmentation", None))

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

        seg_map = img_info["filename"].replace("jpg", "png")

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
        )

        return ann


def find_label_by_name(labels, name):
    matching_labels = [label for label in labels if label.name == name]
    if len(matching_labels) == 1:
        return matching_labels[0]
    elif len(matching_labels) == 0:
        label = LabelEntity(name=name, domain="detection", id=len(labels))
        labels.append(label)
        return label
    else:
        raise ValueError("Found multiple matching labels")


def load_dataset_items_coco_format(
    ann_file_path: str,
    data_root_dir: str,
    subset: Subset = Subset.NONE,
    labels_list: Optional[List[LabelEntity]] = None,
):
    test_mode = subset in {Subset.VALIDATION, Subset.TESTING}

    coco_dataset = CocoDataset(
        ann_file=ann_file_path,
        data_root=data_root_dir,
        classes=None,
        test_mode=test_mode,
    )
    coco_dataset.test_mode = False
    for label_name in coco_dataset.classes:
        find_label_by_name(labels_list, label_name)

    dataset_items = []
    for item in coco_dataset:

        def create_gt_box(x1, y1, x2, y2, label_name):
            return Annotation(
                Rectangle(x1=x1, y1=y1, x2=x2, y2=y2),
                labels=[ScoredLabel(label=find_label_by_name(labels_list, label_name))],
            )

        img_height = item["img_info"].get("height")
        img_width = item["img_info"].get("width")
        divisor = np.array(
            [img_width, img_height, img_width, img_height],
            dtype=item["gt_bboxes"].dtype,
        )
        bboxes = item["gt_bboxes"] / divisor
        labels = item["gt_labels"]

        if item["img_prefix"] is not None:
            filename = osp.join(item["img_prefix"], item["img_info"]["filename"])
        else:
            filename = item["img_info"]["filename"]

        shapes = [
            create_gt_box(x1, y1, x2, y2, coco_dataset.classes[label_id])
            for (x1, y1, x2, y2), label_id in zip(bboxes, labels)
        ]

        dataset_item = DatasetItemEntity(
            media=Image(file_path=filename),
            annotation_scene=AnnotationSceneEntity(
                annotations=shapes, kind=AnnotationSceneKind.ANNOTATION
            ),
            subset=subset,
        )
        dataset_items.append(dataset_item)

    return dataset_items

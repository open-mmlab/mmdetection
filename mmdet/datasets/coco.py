import os.path as osp

import mmcv
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, PolyMaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, show_ann, random_scale
from .utils import DataContainer as DC


def parse_ann_info(ann_info, cat2label, with_mask=True):
    """Parse bbox and mask annotation.

    Args:
        ann_info (list[dict]): Annotation info of an image.
        cat2label (dict): The mapping from category ids to labels.
        with_mask (bool): Whether to parse mask annotations.

    Returns:
        tuple: gt_bboxes, gt_labels and gt_mask_info
    """
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    # each mask consists of one or several polys, each poly is a list of float.
    if with_mask:
        gt_mask_polys = []
        gt_poly_lens = []
    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0 or w < 1 or h < 1:
            continue
        bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
        if ann['iscrowd']:
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_labels.append(cat2label[ann['category_id']])
            if with_mask:
                # Note polys are not resized
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
        ann['mask_polys'] = gt_mask_polys
        ann['poly_lens'] = gt_poly_lens
    return ann


class CocoDataset(Dataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False,
                 debug=False):
        # path of the data file
        self.coco = COCO(ann_file)
        # filter images with no annotation during training
        if not test_mode:
            self.img_ids, self.img_infos = self._filter_imgs()
        else:
            self.img_ids = self.coco.getImgIds()
            self.img_infos = [
                self.coco.loadImgs(idx)[0] for idx in self.img_ids
            ]
        assert len(self.img_ids) == len(self.img_infos)
        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # prefix of images path
        self.img_prefix = img_prefix
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # color channel order and normalize configs
        self.img_norm_cfg = img_norm_cfg
        # proposals
        self.proposals = mmcv.load(
            proposal_file) if proposal_file is not None else None
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        # with crowd or not, False when using RetinaNet
        self.with_crowd = with_crowd
        # with mask or not
        self.with_mask = with_mask
        # with label is False for RPN
        self.with_label = with_label
        # in test mode or not
        self.test_mode = test_mode
        # debug mode or not
        self.debug = debug

        # set group flag for the sampler
        self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = PolyMaskTransform()
        self.numpy2tensor = Numpy2Tensor()

    def __len__(self):
        return len(self.img_ids)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        valid_ids = []
        img_infos = []
        for i in img_ids:
            info = self.coco.loadImgs(i)[0]
            if min(info['width'], info['height']) >= min_size:
                valid_ids.append(i)
                img_infos.append(info)
        return valid_ids, img_infos

    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.img_ids), dtype=np.uint8)
        for i in range(len(self.img_ids)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            img_info = self.img_infos[idx]
            ann_info = self._load_ann_info(idx)

            # load image
            img = mmcv.imread(osp.join(self.img_prefix, img_info['file_name']))
            if self.debug:
                show_ann(self.coco, img, ann_info)

            # load proposals if necessary
            if self.proposals is not None:
                proposals = self.proposals[idx][:self.num_max_proposals, :4]
                # TODO: Handle empty proposals properly. Currently images with
                # no proposals are just ignored, but they can be used for
                # training in concept.
                if len(proposals) == 0:
                    idx = self._rand_another(idx)
                    continue

            ann = parse_ann_info(ann_info, self.cat2label, self.with_mask)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            gt_bboxes_ignore = ann['bboxes_ignore']
            # skip the image if there is no valid gt bbox
            if len(gt_bboxes) == 0:
                idx = self._rand_another(idx)
                continue

            # apply transforms
            flip = True if np.random.rand() < self.flip_ratio else False
            img_scale = random_scale(self.img_scales)  # sample a scale
            img, img_shape, scale_factor = self.img_transform(
                img, img_scale, flip)
            if self.proposals is not None:
                proposals = self.bbox_transform(proposals, img_shape,
                                                scale_factor, flip)
            gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                            flip)
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)

            if self.with_mask:
                gt_mask_polys, gt_poly_lens, num_polys_per_mask = \
                    self.mask_transform(
                        ann['mask_polys'], ann['poly_lens'],
                        img_info['height'], img_info['width'], flip)

            ori_shape = (img_info['height'], img_info['width'], 3)
            img_meta = dict(
                ori_shape=ori_shape,
                img_shape=img_shape,
                scale_factor=scale_factor,
                flip=flip)

            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                gt_bboxes=DC(to_tensor(gt_bboxes)))
            if self.proposals is not None:
                data['proposals'] = DC(to_tensor(proposals))
            if self.with_label:
                data['gt_labels'] = DC(to_tensor(gt_labels))
            if self.with_crowd:
                data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
            if self.with_mask:
                data['gt_masks'] = dict(
                    polys=DC(gt_mask_polys, cpu_only=True),
                    poly_lens=DC(gt_poly_lens, cpu_only=True),
                    polys_per_mask=DC(num_polys_per_mask, cpu_only=True))
            return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['file_name']))
        proposal = (self.proposals[idx][:, :4]
                    if self.proposals is not None else None)

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, scale_factor = self.img_transform(
                img, scale, flip)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                _proposal = self.bbox_transform(proposal, scale_factor, flip)
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        if self.proposals is not None:
            data['proposals'] = proposals
        return data

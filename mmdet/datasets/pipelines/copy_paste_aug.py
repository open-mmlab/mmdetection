import random

import cv2
import numpy as np

from .auto_augment import Rotate
from .transforms import Resize, RandomFlip
from ..builder import PIPELINES


@PIPELINES.register_module()
class CopyPaste:
    def __init__(
        self,
        copy_all=True,
        filter_ignored=False,
        horizontal_flip_ratio=0.5,
        max_rotate_angle=30,
        rotate_level=2,
        rotate_prob=0.5
    ):
        self.copy_all = copy_all
        self.filter_ignored = filter_ignored
        self.resize = None
        self.flip = RandomFlip(flip_ratio=horizontal_flip_ratio)
        if max_rotate_angle > 0:
            self.rotate = Rotate(level=rotate_level, prob=rotate_prob, max_rotate_angle=max_rotate_angle)
        else:
            self.rotate = None
        self.colors = [(np.random.rand(3) * 255).astype(np.uint8) for _ in range(20)]

    def image_copy_paste(self, results, composed_mask):
        img_dtype = results['img'].dtype
        composed_mask = composed_mask[..., None]
        results['img'] = results['copy_paste']['img'] * composed_mask + results['img'] * (1 - composed_mask)
        results['img'] = results['img'].astype(img_dtype)

    def masks_copy_paste(self, results, paste_objects, composed_mask):
        results['gt_masks'].masks = np.array([
            np.where(composed_mask == 1, 0, mask).astype(np.uint8) for mask in results['gt_masks'].masks
        ])
        results['gt_masks'].masks = np.concatenate(
            (results['gt_masks'].masks, results['copy_paste']['gt_masks'].masks[paste_objects]), axis=0)
        results['gt_bboxes'] = np.concatenate(
            (results['gt_bboxes'], results['copy_paste']['gt_bboxes'][paste_objects]), axis=0)
        self.extract_bboxes(results)

    def concatenate_labels(self, results, paste_objects=None):
        results['gt_labels'] = np.concatenate(
            (results['gt_labels'], results['copy_paste']['gt_labels'][paste_objects]), axis=0)
        if not self.filter_ignored:
            results['gt_bboxes_ignore'] = np.concatenate(
                (results['gt_bboxes_ignore'], results['copy_paste']['gt_bboxes_ignore']), axis=0)

    @staticmethod
    def _filter(results, inds):
        results['gt_masks'].masks = results['gt_masks'].masks[inds]
        results['gt_bboxes'] = results['gt_bboxes'][inds]
        results['gt_labels'] = results['gt_labels'][inds]
        if len(results['gt_masks'].masks.shape) == 2:
            results['gt_masks'].masks = np.expand_dims(results['gt_masks'].masks, axis=0)
        if len(results['gt_bboxes'].shape) == 1:
            results['gt_bboxes'] = np.expand_dims(results['gt_bboxes'], axis=0)
        if len(results['gt_labels'].shape) == 0:
            results['gt_labels'] = np.expand_dims(results['gt_labels'], axis=0)

    def filter_empty(self, results, min_size=5):
        w = results['gt_bboxes'][:, 2] - results['gt_bboxes'][:, 0]
        h = results['gt_bboxes'][:, 3] - results['gt_bboxes'][:, 1]
        # Filter too small objects
        inds = np.argwhere((w > min_size) & (h > min_size)).squeeze()
        self._filter(results, inds)
        # Filter objects with empty mask
        inds = np.argwhere(results['gt_masks'].areas > 0).squeeze()
        self._filter(results, inds)

    def filter_ignored_objects(self, results, k=0.9):
        if len(results['gt_bboxes_ignore']) == 0:
            return
        inds = []
        for i, bbox in enumerate(results['gt_bboxes']):
            c = 0
            for ignore in results['gt_bboxes_ignore']:
                x0, x1 = max(bbox[0], ignore[0]), min(bbox[2], ignore[2])
                y0, y1 = max(bbox[1], ignore[1]), min(bbox[3], ignore[3])
                s_intersection = (x1 - x0) * (y1 - y0)
                s_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if s_intersection > k * s_bbox:
                    c += 1
            if c == 0:
                inds.append(i)
        self._filter(results, inds)
        results['gt_bboxes_ignore'] = np.array([])
        results['bbox_fields'] = [field for field in results['bbox_fields'] if field != 'gt_bboxes_ignore']

    @staticmethod
    def cast(results, bbox_type, mask_type, label_type):
        results['gt_masks'].masks = results['gt_masks'].masks.astype(mask_type)
        results['gt_bboxes'] = results['gt_bboxes'].astype(bbox_type)
        results['gt_labels'] = results['gt_labels'].astype(label_type)

    @staticmethod
    def extract_bboxes(results):
        bboxes = []
        for mask in results['gt_masks'].masks:
            xindices = np.where(np.any(mask, axis=0))[0]
            yindices = np.where(np.any(mask, axis=1))[0]
            if yindices.shape[0]:
                x1, x2 = xindices[[0, -1]]
                y1, y2 = yindices[[0, -1]]
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
            bboxes.append((x1, y1, x2, y2))
        results['gt_bboxes'] = np.array(bboxes)
        return bboxes

    def visualize(self, results, name='', draw_ignore=False):
        def draw_rect(img, bbox, color):
            bbox = [int(x) for x in bbox]
            x0, y0, x1, y1 = bbox
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        img = np.copy(results['img'])
        n = max(results['gt_bboxes'].shape[0], results['gt_masks'].masks.shape[0])
        if draw_ignore:
            for bbox in results['gt_bboxes_ignore']:
                draw_rect(img, bbox, (0, 0, 255))
        for i in range(n):
            bbox = results['gt_bboxes'][i] if i < results['gt_bboxes'].shape[0] else None
            mask = results['gt_masks'].masks[i] if i < results['gt_masks'].masks.shape[0] else None
            if bbox is not None:
                draw_rect(img, bbox, (0, 255, 0))
            if mask is not None:
                color = self.colors[i % len(self.colors)]
                colored_mask = np.stack([mask, mask, mask], axis=-1) * color
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0.0)
        cv2.imshow(name, img)
        cv2.waitKey(0)

    def rescale_paste_target(self, results, target_size):
        h, w = target_size
        image_size = results['img'].shape[:-1]
        if image_size != target_size:
            if self.resize is None:
                self.resize = Resize(img_scale=(w, h), keep_ratio=False, multiscale_mode='value')
            else:
                self.resize.img_scale = (w, h)
            results['keep_ratio'] = False
            results['scale'] = (w, h)
            results.pop('scale_factor')
            results = self.resize(results)
        return results

    def get_composed_mask(self, results, h, w, inds=None):
        composed_mask = np.zeros((h, w), dtype=np.uint8)
        for i, mask in enumerate(results['copy_paste']['gt_masks'].masks):
            if inds is None or i in inds:
                composed_mask = np.logical_or(composed_mask, mask)
        return composed_mask

    def __call__(self, results):
        if not 'copy_paste' in results:
            return results
        # Get types of modified objects
        bbox_type = results['gt_bboxes'].dtype
        mask_type = results['gt_masks'].masks.dtype
        label_type = results['gt_labels'].dtype
        # Filter ignored objects
        if self.filter_ignored:
            self.filter_ignored_objects(results['copy_paste'])
        if len(results['copy_paste']['gt_bboxes']) == 0:
            return results
        # Apply augmentations to an image that will share masks
        h, w = results['img'].shape[:-1]
        self.rescale_paste_target(results['copy_paste'], (h, w))
        results['copy_paste'] = self.flip(results['copy_paste'])
        if self.rotate is not None:
            results['copy_paste'] = self.rotate(results['copy_paste'])
        # Get random objects from the image
        objects_num = results['copy_paste']['gt_labels'].shape[0]
        random_num = min(random.randint(0, int(1.1 * objects_num)), objects_num)
        all_nums = list(range(objects_num))
        objects_inds = random.sample(all_nums, random_num) if not self.copy_all else all_nums
        # If list of randomly selected objects is empty do nothing
        if len(objects_inds) == 0:
            return results
        objects_inds = np.array(objects_inds)
        # Get composed mask from N masks on the image
        composed_mask = self.get_composed_mask(results, h, w, objects_inds)
        # Copypaste objects to target image
        self.image_copy_paste(results, composed_mask=composed_mask)
        self.masks_copy_paste(results, objects_inds, composed_mask=composed_mask)
        self.concatenate_labels(results, objects_inds)
        self.filter_empty(results)
        self.cast(results, bbox_type, mask_type, label_type)
        return results

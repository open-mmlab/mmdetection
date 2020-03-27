import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch


class BitMapMasks:

    def __init__(self, masks, height, width):
        """
        Args:
            masks (ndarray): ndarray of masks in shape (N, H, W), where N is
                the number of objects.

        Returns:
            list: a list (length = image num) of list (length = mask num) of
                list (length = poly num) of numpy array
        """
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        return BitMapMasks(self.masks[index], self.height, self.width)

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation='nearest'):
        if len(self.masks) == 0:
            rescaled_masks = np.empty((0, self.height, self.width),
                                      dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                mmcv.imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitMapMasks(rescaled_masks, height, width)

    def resize(self, h, w, interpolation='nearest'):
        if len(self.masks) == 0:
            resized_masks = np.empty((0, ) + (h, w), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                mmcv.imresize(mask, (w, h), interpolation=interpolation)
                for mask in self.masks
            ])
        return BitMapMasks(resized_masks, h, w)

    def flip(self, flip_direction):
        assert flip_direction in ('horizontal', 'vertical')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                mmcv.imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitMapMasks(flipped_masks, self.height, self.width)

    def pad(self, h, w, pad_val):
        if len(self.masks) == 0:
            padded_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                mmcv.impad(mask, (h, w), pad_val=pad_val)
                for mask in self.masks
            ])
        return BitMapMasks(padded_masks, h, w)

    def crop(self, bbox):
        # clip the boundary
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width - 1)
        bbox[1::3] = np.clip(bbox[1::3], 0, self.height - 1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1:y1 + h, x1:x1 + w]
        return BitMapMasks(cropped_masks, h, w)

    def to_tensor(self, dtype, device):
        return torch.tensor(self.masks, dtype=dtype, device=device)


class PolygonMasks:

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        return PolygonMasks([self.masks[index]], self.height, self.width)

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale):
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        return self.resize(new_h, new_w)

    def resize(self, h, w, interpolation=None):
        h_scale = h / self.height
        w_scale = w / self.width
        scaled_masks = []
        for poly_per_obj in self.masks:
            scaled_poly = []
            for p in poly_per_obj:
                p = p.copy()
                p[0::2] *= w_scale
                p[1::2] *= h_scale
                scaled_poly.append(p)
            scaled_masks.append(scaled_poly)
        return PolygonMasks(scaled_masks, h, w)

    def flip(self, flip_direction):
        assert flip_direction in ('horizontal', 'vertical')
        if flip_direction == 'horizontal':
            dim = self.width
            idx = 0
        elif flip_direction == 'vertical':
            dim = self.height
            idx = 1
        flipped_masks = []
        for poly_per_obj in self.masks:
            flipped_poly_per_obj = []
            for p in poly_per_obj:
                p = p.copy()
                p[idx::2] = dim - p[idx::2] - 1
                flipped_poly_per_obj.append(p)
            flipped_masks.append(flipped_poly_per_obj)
        return PolygonMasks(flipped_masks, self.height, self.width)

    def crop(self, bbox):
        # clip the boundary
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width - 1)
        bbox[1::3] = np.clip(bbox[1::3], 0, self.height - 1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        cropped_masks = []
        for poly_per_obj in self.masks:
            cropped_poly_per_obj = []
            for p in poly_per_obj:
                p = p.copy()
                p[0::2] -= bbox[0]
                p[1::2] -= bbox[1]
                cropped_poly_per_obj.append(p)
            cropped_masks.append(cropped_poly_per_obj)

        return PolygonMasks(cropped_masks, h, w)

    def pad(self, *args, **kwargs):
        return self

    def to_bitmap(self):
        bitmap_masks = []
        for poly_per_obj in self.masks:
            rles = maskUtils.frPyObjects(poly_per_obj, self.height, self.width)
            rle = maskUtils.merge(rles)
            bitmap_mask = maskUtils.decode(rle).astype(np.bool)
            bitmap_masks.append(bitmap_mask)
        return BitMapMasks(bitmap_masks, self.height, self.width)

    def to_tensor(self, dtype, device):
        return self.to_bitmap().to_tensor(dtype, device)

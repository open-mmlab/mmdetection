import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch


class Masks(object):

    def rescale(self, scale, interpolation='nearest'):
        pass

    def resize(self, h, w, interpolation='nearest'):
        pass

    def flip(self, flip_direction):
        pass

    def pad(self, h, w, pad_val):
        pass

    def crop(self, bbox):
        pass

    def crop_and_resize(self,
                        bboxes,
                        mask_h,
                        mask_w,
                        assigned_inds,
                        interpolation='bilinear'):
        pass

    def expand(self, expanded_h, expanded_w, top, left):
        pass

    def to_tensor(self, dtype, device):
        pass


class BitMapMasks(Masks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
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

    def __iter__(self):
        return iter(self.masks)

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation='nearest'):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`

        Args:
            scale (tuple[int]): the maximum size
            interpolation (str): same as :func:`mmcv.imrescale`

        Returns:
            BitMapMasks: the rescaled masks
        """
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                mmcv.imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitMapMasks(rescaled_masks, height, width)

    def resize(self, h, w, interpolation='nearest'):
        """Resize masks to the given size of (h, w).

        Args:
            h (int): height of resized mask
            w (int): width of resized mask
            interpolation (str): see `mmcv.imresize`

        Returns:
            BitMapMasks: the resized masks
        """
        if len(self.masks) == 0:
            resized_masks = np.empty((0, ) + (h, w), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                mmcv.imresize(mask, (w, h), interpolation=interpolation)
                for mask in self.masks
            ])
        return BitMapMasks(resized_masks, h, w)

    def flip(self, flip_direction):
        """flip masks alone the given direction.

        Args:
            flip_direction (str): either 'horizontal' or 'vertical'

        Returns:
            BitMapMasks: the flipped masks
        """
        assert flip_direction in ('horizontal', 'vertical')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                mmcv.imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitMapMasks(flipped_masks, self.height, self.width)

    def pad(self, h, w, pad_val=0):
        """Pad masks to the given size of (h, w).

        Args:
            h (int): height of padded mask
            w (int): width of padded mask
            pad_val (int): the padded value

        Returns:
            BitMapMasks: the padded masks
        """
        if len(self.masks) == 0:
            padded_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                mmcv.impad(mask, (h, w), pad_val=pad_val)
                for mask in self.masks
            ])
        return BitMapMasks(padded_masks, h, w)

    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): bbox in format [x1, y1, x2, y2], shape (4, )

        Return:
            BitMapMasks: the cropped masks.
        """
        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height - 1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1:y1 + h, x1:x1 + w]
        return BitMapMasks(cropped_masks, h, w)

    def crop_and_resize(self,
                        bboxes,
                        mask_h,
                        mask_w,
                        assigned_inds,
                        interpolation='bilinear'):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (ndarray): bboxes in format [x1, y1, x2, y2], shape (N, 4)
            mask_h (int): height of resized mask
            mask_w (int): width of resized mask
            assigned_inds (ndarray): indexes to assign masks to each bbox
            interpolation (str): see `mmcv.imresize`

        Return:
            ndarray: the cropped and resized masks.
        """
        if len(self.masks) == 0:
            return np.empty((0, mask_h, mask_w), dtype=np.unit8)

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[assigned_inds[i]]
            bbox = bboxes[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            resized_masks.append(
                mmcv.imresize(
                    mask[y1:y1 + h, x1:x1 + w], (mask_w, mask_h),
                    interpolation=interpolation))
        return np.stack(resized_masks)

    def expand(self, expanded_h, expanded_w, top, left):
        """see `transforms.Expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w),
                                     dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w),
                                     dtype=np.uint8)
            expanded_mask[:, top:top + self.height,
                          left:left + self.width] = self.masks
        return BitMapMasks(expanded_mask, expanded_h, expanded_w)

    def to_tensor(self, dtype, device):
        return torch.from_numpy(self.masks, dtype=dtype, device=device)


class PolygonMasks(Masks):
    """This class represents masks in the form of polygons.

    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates

    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    'Unsupported input of type {} for indexing!'.format(
                        type(index)))
        if isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """see BitMapMasks.rescale"""
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize(new_h, new_w)
        return rescaled_masks

    def resize(self, h, w, interpolation=None):
        """see BitMapMasks.resize"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], h, w)
        else:
            h_scale = h / self.height
            w_scale = w / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] *= w_scale
                    p[1::2] *= h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, h, w)
        return resized_masks

    def flip(self, flip_direction):
        """see BitMapMasks.flip"""
        assert flip_direction in ('horizontal', 'vertical')
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            if flip_direction == 'horizontal':
                dim = self.width
                idx = 0
            elif flip_direction == 'vertical':
                dim = self.height
                idx = 1
            else:
                raise NotImplementedError
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[idx::2] = dim - p[idx::2] - 1
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height,
                                         self.width)
        return flipped_masks

    def crop(self, bbox):
        """see BitMapMasks.crop"""
        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height - 1)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            cropped_masks = []
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] -= bbox[0]
                    p[1::2] -= bbox[1]
                    cropped_poly_per_obj.append(p)
                cropped_masks.append(cropped_poly_per_obj)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, *args, **kwargs):
        """padding has no effect on polygons"""
        return self

    def expand(self, *args, **kwargs):
        raise NotImplementedError

    def crop_and_resize(self,
                        bboxes,
                        mask_h,
                        mask_w,
                        assigned_inds,
                        interpolation='bilinear'):
        """see BitMapMasks.crop_and_resize"""
        if len(self.masks) == 0:
            return np.empty((0, mask_h, mask_w), dtype=np.unit8)

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[assigned_inds[i]]
            bbox = bboxes[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            h_scale = mask_h / h
            w_scale = mask_w / w

            resized_mask = []
            for p in mask:
                p = p.copy()
                # crop
                p[0::2] -= bbox[0]
                p[1::2] -= bbox[1]

                # resize
                p[0::2] *= w_scale
                p[1::2] *= h_scale
                resized_mask.append(p)
            resized_masks.append(
                polygon_to_bitmap(resized_mask, mask_h, mask_w))

        return np.stack(resized_masks)

    def to_bitmap(self):
        """convert polygon masks to bitmap masks"""
        if len(self.masks) == 0:
            return BitMapMasks(
                np.empty((0, self.height, self.width), dtype=np.uint8),
                self.height, self.width)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width))
        return BitMapMasks(bitmap_masks, self.height, self.width)

    def to_tensor(self, dtype, device):
        bitmap_masks = polygon_to_bitmap(self.masks, self.height,
                                         self.width).to_tensor(dtype, device)
        return torch.from_numpy(bitmap_masks, dtype=dtype, device=device)


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

        Args:
            polygons (list[ndarray]): masks in polygon representation
            height (int): mask height
            width (int): mask width

        Return:
            ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(np.bool)
    return bitmap_mask

from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch


class BaseInstanceMasks(metaclass=ABCMeta):

    @abstractmethod
    def rescale(self, scale, interpolation='nearest'):
        pass

    @abstractmethod
    def resize(self, out_shape, interpolation='nearest'):
        pass

    @abstractmethod
    def flip(self, flip_direction='horizontal'):
        pass

    @abstractmethod
    def pad(self, out_shape, pad_val):
        pass

    @abstractmethod
    def crop(self, bbox):
        pass

    @abstractmethod
    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        interpolation='bilinear'):
        pass

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        pass

    @abstractmethod
    def to_ndarray(self):
        pass

    @abstractmethod
    def to_tensor(self, dtype, device):
        pass


class BitMapMasks(BaseInstanceMasks):
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
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitMapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __len__(self):
        return len(self.masks)

    def rescale(self, scale, interpolation='nearest'):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`

        Args:
            scale (tuple[int]): the maximum size (h, w) of rescaled mask
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

    def resize(self, out_shape, interpolation='nearest'):
        """Resize masks to the given out_shape.

        Args:
            out_shape: target (h, w) of resized mask
            interpolation (str): see `mmcv.imresize`

        Returns:
            BitMapMasks: the resized masks
        """
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                mmcv.imresize(mask, out_shape, interpolation=interpolation)
                for mask in self.masks
            ])
        return BitMapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction='horizontal'):
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

    def pad(self, out_shape, pad_val=0):
        """Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): target (h, w) of padded mask
            pad_val (int): the padded value

        Returns:
            BitMapMasks: the padded masks
        """
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                mmcv.impad(mask, out_shape, pad_val=pad_val)
                for mask in self.masks
            ])
        return BitMapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): bbox in format [x1, y1, x2, y2], shape (4, )

        Return:
            BitMapMasks: the cropped masks.
        """
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

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
                        out_shape,
                        inds,
                        interpolation='bilinear'):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (ndarray): bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): target (h, w) of resized mask
            inds (ndarray): indexes to assign masks to each bbox
            interpolation (str): see `mmcv.imresize`

        Return:
            ndarray: the cropped and resized masks.
        """
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitMapMasks(empty_masks, *out_shape)

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            resized_masks.append(
                mmcv.imresize(
                    mask[y1:y1 + h, x1:x1 + w],
                    out_shape,
                    interpolation=interpolation))
        return BitMapMasks(np.stack(resized_masks), *out_shape)

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

    def to_ndarray(self):
        return self.masks

    def to_tensor(self, dtype, device):
        return torch.tensor(self.masks, dtype=dtype, device=device)


class PolygonMasks(BaseInstanceMasks):
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
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

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
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """see BitMapMasks.resize"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] *= w_scale
                    p[1::2] *= h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction='horizontal'):
        """see BitMapMasks.flip"""
        assert flip_direction in ('horizontal', 'vertical')
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            if flip_direction == 'horizontal':
                dim = self.width
                idx = 0
            else:
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
            flipped_masks = PolygonMasks(flipped_masks, self.height,
                                         self.width)
        return flipped_masks

    def crop(self, bbox):
        """see BitMapMasks.crop"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

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

    def pad(self, out_shape, pad_val=0):
        """padding has no effect on polygons"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        raise NotImplementedError

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        interpolation='bilinear'):
        """see BitMapMasks.crop_and_resize"""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            h_scale = out_h / h
            w_scale = out_w / w

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
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def to_bitmap(self):
        """convert polygon masks to bitmap masks"""
        bitmap_masks = self.to_ndarray()
        return BitMapMasks(bitmap_masks, self.height, self.width)

    def to_ndarray(self):
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width))
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width),
                               dtype=dtype,
                               device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)


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

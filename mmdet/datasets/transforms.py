import mmcv
import numpy as np
import torch

from mmdet.core.mask_ops import segms

__all__ = [
    'ImageTransform', 'BboxTransform', 'PolyMaskTransform', 'Numpy2Tensor'
]


class ImageTransform(object):
    """Preprocess an image
    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False):
        img, scale_factor = mmcv.imrescale(img, scale, True)
        img_shape = img.shape
        img = mmcv.imnorm(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        img = img.transpose(2, 0, 1)
        return img, img_shape, scale_factor

        # img, scale = cvb.resize_keep_ar(img_or_path, max_long_edge,
        #                                 max_short_edge, True)
        # shape_scale = np.array(img.shape + (scale, ), dtype=np.float32)
        # if flip:
        #     img = img[:, ::-1, :].copy()
        # if self.color_order == 'RGB':
        #     img = cvb.bgr2rgb(img)
        # img = img.astype(np.float32)
        # img -= self.color_mean
        # img /= self.color_std
        # if self.size_divisor is None:
        #     padded_img = img
        # else:
        #     pad_h = int(np.ceil(
        #         img.shape[0] / self.size_divisor)) * self.size_divisor
        #     pad_w = int(np.ceil(
        #         img.shape[1] / self.size_divisor)) * self.size_divisor
        #     padded_img = cvb.pad_img(img, (pad_h, pad_w), pad_val=0)
        # padded_img = padded_img.transpose(2, 0, 1)
        # return padded_img, shape_scale


class ImageCrop(object):
    """crop image patches and resize patches into fixed size
    1. (read and) flip image (if needed)
    2. crop image patches according to given bboxes
    3. resize patches into fixed size (default 224x224)
    4. normalize the image (if needed)
    5. transpose to (c, h, w) (if needed)
    """

    def __init__(self,
                 normalize=True,
                 transpose=True,
                 color_order='RGB',
                 color_mean=(0, 0, 0),
                 color_std=(1, 1, 1)):
        self.normalize = normalize
        self.transpose = transpose

        assert color_order in ['RGB', 'BGR']
        self.color_order = color_order
        self.color_mean = np.array(color_mean, dtype=np.float32)
        self.color_std = np.array(color_std, dtype=np.float32)

    def __call__(self,
                 img_or_path,
                 bboxes,
                 crop_size,
                 scale_ratio=1.0,
                 flip=False):
        img = cvb.read_img(img_or_path)
        if flip:
            img = img[:, ::-1, :].copy()
        crop_imgs = cvb.crop_img(
            img,
            bboxes[:, :4],
            scale_ratio=scale_ratio,
            pad_fill=self.color_mean)
        processed_crop_imgs_list = []
        for i in range(len(crop_imgs)):
            crop_img = crop_imgs[i]
            crop_img = cvb.resize(crop_img, crop_size)
            crop_img = crop_img.astype(np.float32)
            crop_img -= self.color_mean
            crop_img /= self.color_std
            processed_crop_imgs_list.append(crop_img)
        processed_crop_imgs = np.stack(processed_crop_imgs_list, axis=0)
        processed_crop_imgs = processed_crop_imgs.transpose(0, 3, 1, 2)
        return processed_crop_imgs


class BboxTransform(object):
    """Preprocess gt bboxes
    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = mmcv.bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class PolyMaskTransform(object):

    def __init__(self):
        pass

    def __call__(self, gt_mask_polys, gt_poly_lens, img_h, img_w, flip=False):
        """
        Args:
            gt_mask_polys(list): a list of masks, each mask is a list of polys,
                each poly is a list of numbers
            gt_poly_lens(list): a list of int, indicating the size of each poly
        """
        if flip:
            gt_mask_polys = segms.flip_segms(gt_mask_polys, img_h, img_w)
        num_polys_per_mask = np.array(
            [len(mask_polys) for mask_polys in gt_mask_polys], dtype=np.int64)
        gt_poly_lens = np.array(gt_poly_lens, dtype=np.int64)
        gt_mask_polys = [
            np.concatenate(mask_polys).astype(np.float32)
            for mask_polys in gt_mask_polys
        ]
        gt_mask_polys = np.concatenate(gt_mask_polys)
        return gt_mask_polys, gt_poly_lens, num_polys_per_mask


class MaskTransform(object):
    """Preprocess masks
    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __init__(self, max_num_gts, pad_size=None):
        self.max_num_gts = max_num_gts
        self.pad_size = pad_size

    def __call__(self, masks, img_size, flip=False):
        max_long_edge = max(img_size)
        max_short_edge = min(img_size)
        masks = [
            cvb.resize_keep_ar(
                mask,
                max_long_edge,
                max_short_edge,
                interpolation=cvb.INTER_NEAREST) for mask in masks
        ]
        masks = np.stack(masks, axis=0)
        if flip:
            masks = masks[:, ::-1, :]
        if self.pad_size is None:
            pad_h = masks.shape[1]
            pad_w = masks.shape[2]
        else:
            pad_size = self.pad_size if self.pad_size > 0 else max_long_edge
            pad_h = pad_w = pad_size
        padded_masks = np.zeros(
            (self.max_num_gts, pad_h, pad_w), dtype=masks.dtype)
        padded_masks[:masks.shape[0], :masks.shape[1], :masks.shape[2]] = masks
        return padded_masks


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])

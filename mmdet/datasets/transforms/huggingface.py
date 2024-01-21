# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image, ImageOps

from mmdet.datasets.transforms import LoadAnnotations, LoadImageFromNDArray
from mmdet.registry import TRANSFORMS


def _get_dict_value(data: Dict[str, Any], key: Union[str, List[str]]):
    if isinstance(key, str):
        key = key.split('.')
    for k in key:
        data = data[k]
    return data


# Copy from mmcv
# https://github.com/open-mmlab/mmcv/blob/c7c02a7c5ba25d37c7f7928013ffb3016e254eb9/mmcv/image/io.py#L87-L141
def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


@TRANSFORMS.register_module()
class LoadImageFromHuggingface(BaseTransform):

    def __init__(self,
                 img_prefix='image',
                 color_mode='color',
                 to_float32=False):
        super().__init__()

        self.img_prefix = img_prefix

        self.color_mode = color_mode
        self.to_float32 = to_float32

        self._loader = LoadImageFromNDArray(to_float32=to_float32, )

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img = _get_dict_value(results, self.img_prefix)
        img = _pillow2array(img, flag=self.color_mode)

        results['img'] = img
        results['width'] = img.shape[1]
        results['height'] = img.shape[0]

        return self._loader.transform(results)


@TRANSFORMS.register_module()
class LoadAnnotationsFromHuggingface(BaseTransform):

    def __init__(self,
                 bbox_label_prefix='objects.category',
                 bbox_prefix=None,
                 mask_prefix=None,
                 ignore_flag_prefix=None,
                 seg_map_prefix=None,
                 with_mask: bool = False,
                 poly2mask: bool = True,
                 box_type: str = 'hbox',
                 reduce_zero_label: bool = False,
                 ignore_index: int = 255,
                 **kwargs):
        super().__init__()
        self.bbox_prefix = bbox_prefix
        self.bbox_label_prefix = bbox_label_prefix
        self.mask_prefix = mask_prefix
        self.ignore_flag_prefix = ignore_flag_prefix
        self.seg_map_prefix = seg_map_prefix

        self._load_annotations = LoadAnnotations(
            with_mask=with_mask,
            poly2mask=poly2mask,
            box_type=box_type,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        instances = []

        weight = results['width']
        height = results['height']

        try:
            label_list = _get_dict_value(results, self.bbox_label_prefix)
        except KeyError:
            raise ValueError(
                f'No key {self.bbox_label_prefix} found in results')

        if self.bbox_prefix is not None:
            bbox_list = _get_dict_value(results, self.bbox_prefix)
        else:
            bbox_list = None

        if self.mask_prefix is not None:
            mask_list = _get_dict_value(results, self.mask_prefix)
        else:
            mask_list = None

        if self.ignore_flag_prefix is not None:
            ignore_flag_list = _get_dict_value(results,
                                               self.ignore_flag_prefix)
        else:
            ignore_flag_list = None

        for i, label in enumerate(label_list):
            instance = {'bbox_label': label}

            if bbox_list is not None:
                bbox = bbox_list[i]
                x1, y1, w, h = bbox
                inter_w = max(0, min(x1 + w, weight) - max(x1, 0))
                inter_h = max(0, min(y1 + h, height) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]
                instance['bbox'] = bbox

            if mask_list is not None:
                mask = mask_list[i]
                instance['mask'] = mask

            if ignore_flag_list is not None:
                ignore_flag = ignore_flag_list[i]
                instance['ignore_flag'] = ignore_flag
            else:
                instance['ignore_flag'] = 0

            instances.append(instance)

        results['instances'] = instances

        return self._load_annotations.transform(results)

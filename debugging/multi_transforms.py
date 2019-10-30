"""Multi Transforms.

Applies the same transformation on a COCO dataset image and annotation pair.
"""
from PIL import Image
import random

import torchvision.transforms.functional as F
from torchvision.transforms.transforms import Normalize
import numpy as np


__all__ = ["MultiCompose", "MultiToTensor", "MultiNormalize", "MultiResize",
           "MultiRandomFlip", "MultiToPILImage"]


class MultiCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to
            compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, annotation):
        for t in self.transforms:
            img, annotation = t(img, annotation)
        return img, annotation

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class MultiToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to preds.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes
    (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has
    dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    Returns:
        tuple: Tuple (image, target). target is the object returned by
            coco.loadAnns
    """

    def __call__(self, pic, target):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to preds.
            target (object array): Annotation array as returned by
                coco.loadAnns.

        Returns:
            tuple: Tuple of the converted image and the annotation.
        """
        # Nothing happens to the target, so we return it as is
        return F.to_tensor(pic), target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MultiResize(object):
    def __init__(self, size):
        """Resizes the given PIL Image and its corresponsding target.

        Args:
            size (tuple): The size to resize to in (h, w)
        """
        self.h, self.w = size
        self.area = self.h * self.w

    def __call__(self, img, target):
        """Resizes the image and target to the same size.

        Args:
            img (PIL.Image.Image): A PIL Image.
            target (object array): The annotation array.
        """
        img_w, img_h = img.size
        w_ratio = float(self.w) / float(img_w)
        h_ratio = float(self.h) / float(img_h)
        area_ratio = self.area / (img_w * img_h)

        img = img.resize((self.w, self.h), Image.BICUBIC)

        out_target = []

        # Target contains a list of bbox annotations. We have to iterate through
        # all the target objects and resize the properties of each.
        for box in target:
            current_box = {'id': box['id'],
                           'iscrowd': box['iscrowd'],
                           'image_id': box['image_id'],
                           'category_id': box['category_id'],
                           'area': box['area'] * area_ratio}
            # Deal with first case: Not crowd
            if box['iscrowd'] == 0:
                current_segmentation = []
                # Do segmentation first
                # multiply x values by width ratio, y values by height ratio
                for segmentation in box['segmentation']:
                    count = 0
                    current_coordinates = []
                    for coords in segmentation:
                        if count % 2 == 0:
                            current_coordinates.append(coords * w_ratio)
                        else:
                            current_coordinates.append(coords * h_ratio)
                        count += 1
                    current_segmentation.append(current_coordinates)
                current_box['segmentation'] = current_segmentation

            else:
                raise NotImplementedError

            # Next do bboxes
            current_bbox = []
            count = 0
            for coord in box['bbox']:
                if count % 2 == 0:
                    current_bbox.append(coord * w_ratio)
                else:
                    current_bbox.append(coord * h_ratio)
                count += 1
            current_box['bbox'] = current_bbox
            out_target.append(current_box)

        return img, out_target

    def __repr__(self):
        return self.__class__.__name__ + '(size=({}, {})'.format(self.w, self.h)


class MultiRandomFlip:
    def __init__(self, probability):
        """Randomly flips vertically or horizontally with a given probability.

        Args:
            probability (float): Probability to flip.
        """
        self.probability = probability

    def __call__(self, img, target):
        """Flips an image and its target.

        Args:
            img (PIL.Image.Image): A PIL Image.
            target (object array): The annotation array.
        """
        should_flip = np.random.choice(
            np.array([True, False]),
            p=np.array([self.probability, 1 - self.probability]))
        if not should_flip:
            return img, target

        method = random.choice((Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM))

        out_target = []

        # Transform the target
        for box in target:
            current_box = {'id': box['id'],
                           'iscrowd': box['iscrowd'],
                           'image_id': box['image_id'],
                           'category_id': box['category_id'],
                           'area': box['area']}
            # Deal with first case: Not crowd
            if box['iscrowd'] == 0:
                current_segmentation = []
                # Do segmentation first
                # multiply x values by width ratio, y values by height ratio
                for segmentation in box['segmentation']:
                    count = 0
                    current_coordinates = []
                    for coords in segmentation:
                        if count % 2 == 0:
                            # Horizontal transform
                            if method == Image.FLIP_LEFT_RIGHT:
                                current_coordinates.append(img.size[0] - coords)
                            else:
                                current_coordinates.append(coords)
                        else:
                            if method == Image.FLIP_TOP_BOTTOM:
                                current_coordinates.append(img.size[1] - coords)
                            else:
                                current_coordinates.append(coords)
                        count += 1
                    current_segmentation.append(current_coordinates)
                current_box['segmentation'] = current_segmentation

            else:
                raise NotImplementedError

            # Next do bboxes
            x_pos, y_pos, width, height = box['bbox']
            if method == Image.FLIP_LEFT_RIGHT:
                x_pos = img.size[0] - x_pos - width
            else:
                y_pos = img.size[1] - y_pos - height
            current_box['bbox'] = [x_pos, y_pos, width, height]
            out_target.append(current_box)

        return img.transpose(method), out_target


class MultiNormalize:
    def __init__(self, mean, std, **kwargs):
        """Normalizes images.

        Args:
            mean (list): List of means of each channel.
            std (list): List of standard deviations of each channel
        """
        self.normalize = Normalize(mean, std)

    def __call__(self, img, target):
        """Flips an image and its target.

        Args:
            img (PIL.Image.Image): A PIL Image.
            target (object array): The annotation array.
        """
        return self.normalize(img), target


class MultiToPILImage:
    def __call__(self, img, ann):
        """Converts a preds to a PIL Image."""
        return F.to_pil_image(img), ann

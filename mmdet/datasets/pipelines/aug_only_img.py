import cv2
import numpy as np

from mmdet.datasets import PIPELINES

_MAX_LEVEL = 10


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


def random_negative(value, random_negative_prob):
    """Randomly negative value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value


def blend(image1, image2, factor):
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2
    # Do addition in float.
    blended = image1 + factor * (image2 - image1)
    # Interpolate
    if factor > 0. and factor < 1.:
        return blended
    # Extrapolate:
    return np.clip(blended, 0, 255)


@PIPELINES.register_module()
class Color(object):
    """Apply Shear Transformation to image (and its corresponding bbox, mask,
    segmentation).

    Args:
        level (int | float): The level for Translate and should be in
            range (0,_MAX_LEVEL]. This value controls the offset used for
            translate the image/bboxes/masks/seg along with x-axis or y-axis.
        img_fill_val (int | float | tuple): The filled values for image border.
            If float, the same fill value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
        seg_ignore_label (int): The ``img_fill_val`` used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Default 255.
        prob (float): The probability for perform translating and should be in
            range 0 to 1.
        axis (str): Translate images along with x-axis or y-axis. The option
            of axis is 'x' or 'y'.
        max_shear_magnitude (float): The maximum magnitude for shear
            transformation.
    """

    def __init__(
        self,
        level,
        prob=0.5,
    ):
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _color_img(self, results, factor):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            # NOTE convert image from BGR to GRAY
            gray_img = cv2.cvtColor(
                img.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
            gray_img = np.tile(gray_img[..., None],
                               [1, 1, 3]).astype(img.dtype)
            results[key] = blend(gray_img, img, factor).astype(img.dtype)

    def __call__(self, results, random_negative_prob=0.5):
        if np.random.rand() > self.prob:
            return results
        factor = random_negative(self.factor, random_negative_prob)
        self._color_img(results, factor)
        return results

    def __repr__(self, ):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Sharpness(object):

    def __init__(self, level, ksize, sigmaX=0, sigmaY=0, prob=0.5):
        self.level = level
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _sharp_img(self, results, factor):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_transformed = cv2.GaussianBlur(
                img, self.ksize, self.sigmaX, sigmaY=self.sigmaY)
            results[key] = blend(img_transformed, img,
                                 factor).astype(img.dtype)

    def __call__(self, results, random_negative_prob=0.5):
        if np.random.rand() > self.prob:
            return results
        factor = random_negative(self.factor, random_negative_prob)
        self._sharp_img(results, factor)
        return results

    def __repr__(self, ):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'ksize={self.ksize}, '
        repr_str += f'sigmaX={self.sigmaX}, '
        repr_str += f'sigmaY={self.sigmaY}, '
        repr_str += f'prob={self.prob})'

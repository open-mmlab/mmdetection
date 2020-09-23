import mmcv
import numpy as np

from mmdet.datasets import PIPELINES

_MAX_LEVEL = 10


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


@PIPELINES.register_module()
class ColorTransform(object):
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        for key in results.get('img_fields', ['img']):
            # NOTE defaultly the image should be BGR format
            img = results[key]
            results[key] = mmcv.adjust_color(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_color_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class EqualizeTransform(object):
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.prob = prob

    def _imequalize(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.imequalize(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._imequalize(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'


@PIPELINES.register_module()
class BrightnessTransform(object):
    """Apply Brightness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Brightness transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_brightness_img(self, results, factor=1.0):
        """Adjust the brightness of image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_brightness(img,
                                                  factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_brightness_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class ContrastTransform(object):
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _adjust_contrast_img(self, results, factor=1.0):
        """Adjust the image contrast."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.adjust_contrast(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_contrast_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str

import cv2
import numpy as np

from mmdet.datasets import PIPELINES

_MAX_LEVEL = 10


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0. A value of 1.0 means only image2 is used. A value
    between 0.0 and 1.0 means we linearly interpolate the pixel values between
    the two images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values between 0
    and 255.
    """
    if factor == 1.0:
        return image2
    # Do addition in float.
    blended = image1 + factor * (image2 - image1)
    # Interpolate
    if factor > 0. and factor < 1.:
        return blended
    # Extrapolate
    return np.clip(blended, 0, 255)


@PIPELINES.register_module()
class Color(object):
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
            'The level used for calculating Translate\'s offset should be ' \
            'in range [0, _MAX_LEVEL]'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range 0 to 1.'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _color_img(self, results, factor):
        """Apply Color transformation to image."""
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            # NOTE convert image from BGR to GRAY
            gray_img = cv2.cvtColor(
                img.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
            gray_img = np.tile(gray_img[..., None],
                               [1, 1, 3]).astype(img.dtype)
            results[key] = blend(gray_img, img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.rand() > self.prob:
            return results
        self._color_img(results, self.factor)
        return results

    def __repr__(self, ):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class Sharpness(object):
    """Apply Sharpness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): The level for Sharpness. Should be
            in range [0, _MAX_LEVEL].
        ksize (int | tuple): Gaussian kernel size with format (w, h).
            Elements should be positive and odd. Or, they can be zero’s
            and then they are computed from sigma*. Same in 'cv2.GaussianBlur'.
        sigmaX (int | float): Gaussian kernel standard deviation in
            X direction. Same in 'cv2.GaussianBlur'.
        sigmaY (int | float): Gaussian kernel standard deviation in
            Y direction. If 'sigmaY' is zero, it is set to be equal
            to sigmaX. If both sigmas are zeros, they are computed
            from 'ksize'. Same in 'cv2.GaussianBlur'.
        prob (float): The probability for performing Sharpness
            transformation.
    """

    def __init__(self, level, ksize=(5, 5), sigmaX=0, sigmaY=0, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range (0, _MAX_LEVEL]'
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        elif isinstance(ksize, tuple):
            assert len(ksize) == 2, \
                'ksize as tuple must have 2 elements.'
            ksize = tuple([int(size) for size in ksize])
        else:
            raise ValueError(
                'ksize must be int or tuple with 2 elements of int.')
        assert np.all([size % 2 == 1 for size in ksize]) \
            and np.all([size > 0 for size in ksize]), \
            'all elements of ksize must be positive and odd.'
        if not sigmaY:
            sigmaY = sigmaX
        assert isinstance(sigmaX, (int, float)) \
            and isinstance(sigmaY, (int, float)), \
            'sigmaX and sigmaY must be type int or float.'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range 0 to 1.'
        self.level = level
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        self.prob = prob
        self.factor = enhance_level_to_value(level)

    def _sharp_img(self, results, factor):
        """Apply Sharpness to image."""
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_transformed = cv2.GaussianBlur(
                img, self.ksize, self.sigmaX, sigmaY=self.sigmaY)
            results[key] = blend(img_transformed, img,
                                 factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Sharpness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._sharp_img(results, self.factor)
        return results

    def __repr__(self, ):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'ksize={self.ksize}, '
        repr_str += f'sigmaX={self.sigmaX}, '
        repr_str += f'sigmaY={self.sigmaY}, '
        repr_str += f'prob={self.prob})'


@PIPELINES.register_module()
class Equalize(object):
    """Apply Equalize transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        prob (float): The probability for performing Equalize transformation.
    """

    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range 0 to 1.'
        self.prob = prob

    def _equalize_img(self, results):
        """Equalizes the histogram of one image."""
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # equalize the histogram of the Y channel
            img_yuv[..., 0] = cv2.equalizeHist(img_yuv[..., 0])
            # convert the YUV image back to BGR format
            results[key] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def __call__(self, results):
        """Call function for Equalize transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._equalize_img(results)
        return results

    def __repr__(self, ):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'

"""Color Mapping.

Maps colors of a given value to an RGB value.
"""
import numpy as np


color_mapping = ('321713',
                 '401E1F',
                 '4D262E',
                 '58303E',
                 '603B50',
                 '654862',
                 '665675',
                 '636686',
                 '5B7595',
                 '5186A0',
                 '4496A8',
                 '39A5AC',
                 '37B5AC',
                 '43C4A7',
                 '5AD1A0',
                 '77DE96',
                 '98EA8A',
                 'BCF47F',
                 'E2FD76')


def map_color_values(array, n, source_dest=None):
    """Maps values to RGB arrays.

    Shape:
        array: (h, w)

    Args:
        array (np.array): Array of values to map to colors.
        n (int or float): Number of categories to map.
        source (list or tuple): Hex value of source (i.e. 0th category)
    """
    out = np.empty((array.shape[0], array.shape[1], 3), dtype='uint8')
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i][j] = map_color_value(array[i][j], n)

    return out.astype('uint8')


def map_color_value(value, n):
    """Converts colors.

    Maps a color between a value on the interval [0, n] to rgb values. Based
    on HSL. We choose a color by mapping the value x as a fraction of n to a
    value for hue on the interval [0, 360], with 0 = 0 and 1 = 360. This is
    then mapped using a standard HSL to RGB conversion with parameters S = 1,
    L = 0.5.

    Args:
        value (int or float): The value to be mapped. Must be in the range
            0 <= value <= n. If value = n, it is converted to 0.
        n (int or float): The maximum value corresponding to a hue of 360.

    Returns:
        np.array: a numpy array representing RGB values.
    """
    if n == 20:
        # Use nicer color values
        # First convert value to int
        value = int(value)

        if value == 0:
            return np.array([255, 255, 255]).astype('uint8')
        h = color_mapping[value - 1]
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype='uint8')

    multiplier = 360 / n
    if value == n:
        value = 0

    hue = float(value) * float(multiplier)

    c = 1.
    x = 1 - (abs((hue / 60.) % 2. - 1.))

    if 0 <= hue < 60:
        out = np.array([c, x, 0.])
    elif 60 <= hue < 120:
        out = np.array([x, c, 0])
    elif 120 <= hue < 180:
        out = np.array([0, c, x])
    elif 180 <= hue < 240:
        out = np.array([0, x, c])
    elif 240 <= hue < 300:
        out = np.array([x, 0, c])
    else:
        out = np.array([c, 0, x])

    return (out * 255).astype('uint8')


def map_alpha_values(array, color, n):
    """Maps energy or centerness values to red with RGBA output.

    Args:
        array (np.array): Array of values to be converted in the range [0, n].
        color (np.array): Array of colors in RGB in the range [0, 255].
        n (int or float): Maximum value that corresponds to 100% opacity.

    Returns:
        np.array: a numpy array representing RGBA values.
    """
    out = np.empty((array.shape[0], array.shape[1], 4))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i][j] = map_alpha_value(array[i][j], color, n)

    return out.astype('uint8')


def map_alpha_value(value, color, n):
    """Maps a single value to the alpha parameter of an RGBA array."""
    return np.array([color[0], color[1], color[2], (value / n) * 255])


def map_bool_values(array, color):
    """Maps boolean values to a certain color with RGBA output."""
    out = np.empty((array.shape[0], array.shape[1], 4))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i][j] = map_bool_value(array[i][j], color)
    return out.astype('uint8')


def map_bool_value(value, color):
    """Maps a single value to the alpha parameter of an RGBA array."""
    a = 255 if value else 0
    return np.array([color[0], color[1], color[2], a])
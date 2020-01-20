import numpy as np
from PIL import Image, ImageDraw


COLOR_MAPPING = ('24100E',
                 '321713',
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


def colorize_class_preds(class_maps, no_classes):
    # class maps are level-batch-class-H-W
    np_arrays = []
    for lvl in class_maps:
        lvl = map_color_values(lvl, no_classes)
        np_arrays.append(lvl)

    return np_arrays


def normalize_centerness(center_maps):

    p_min = 1E6
    p_max = -1E6
    for lvl in center_maps:
        p_min = np.min([p_min, np.min(lvl)])
        p_max = np.max([p_max, np.max(lvl)])

    normed_imgs = []
    for lvl in center_maps:
        lvl = (lvl - p_min) / (p_max - p_min) * 255
        normed_imgs.append(lvl)

    return normed_imgs


def image_pyramid(pred_maps, target_size):
    resized_imgs = []
    for lvl in pred_maps:
        lvl = lvl.astype(np.uint8)
        lvl_img = Image.fromarray(lvl)
        lvl_img = lvl_img.resize(target_size[::-1])
        lvl_img = np.array(lvl_img)
        resized_imgs.append(lvl_img)
        resized_imgs.append(np.zeros((10,) + lvl_img.shape[1:]))
    img_cat = np.concatenate(resized_imgs)
    return img_cat.astype(np.uint8)

def get_present_classes(classes_vis):
    unique_vals = []
    for vis in classes_vis:
        if isinstance(vis,np.ndarray):
            unique_vals.extend(np.unique(vis))
        else:
            unique_vals.extend(np.unique(vis.cpu().numpy()))

    ret = set(unique_vals)
    try:
        ret.remove(-1)
    except KeyError:
        pass
    ret = list(ret)
    ret.sort()
    return ret

def stitch_big_image(images_list):
    if isinstance(images_list[0], np.ndarray):
        # stitch vertically
        # stack to 3 channels if necessary
        max_len = 0
        for ind, ele in enumerate(images_list):
            if ele.shape[-1] == 1:
                images_list[ind] = np.concatenate([ele, ele, ele],-1)
            if ele.shape[1] > max_len:
                max_len = ele.shape[1]
        for ind, ele in enumerate(images_list):
            if ele.shape[1]<max_len:
                pad_ele = np.zeros((ele.shape[0],max_len-ele.shape[1],3),np.uint8)
                images_list[ind] = np.concatenate([pad_ele,images_list[ind]], 1)

        return np.concatenate(images_list,0)
    else:
        # stitch horizontally
        stich_list = [stitch_big_image(im) for im in images_list]

    return np.concatenate(stich_list, 1)


def add_class_legend(img, classes, present_classes):
    max_len = max([len(x) for x in classes])
    no_cl = len(classes)

    spacer = 20
    canv = np.ones((img.shape[0], 25 + max_len * 7, 3)) * 255


    for ind, cla in enumerate(present_classes):
        col_block = map_color_values(np.ones((10, 10)) * cla, no_cl)
        canv[ind * spacer + 10:ind * spacer + 20, 10:20] = col_block
    canv_img = Image.fromarray(canv.astype(np.uint8))
    draw = ImageDraw.Draw(canv_img)

    for ind, cla in enumerate(present_classes):
        try:
            label = classes[cla]
        except IndexError:
            label = 'Unknown Class'
        draw.text((25, ind * spacer + 10), label, (0, 0, 0))

    canv = np.array(canv_img).astype(np.uint8)

    return np.concatenate((canv, img), axis=1)


def map_color_values(array, n):
    """Maps values to RGB arrays.

    Shape:
        array: (h, w)

    Args:
        array (np.ndarray): Array of values to map to colors.
        n (int or float): Number of categories to map.
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
        np.ndarray: a numpy array representing RGB values.
    """
    if value < 0:
        return np.array([0, 0, 0]).astype(np.uint8)

    if value == n:
        value = 0

    if n == 20:
        # Use nicer color values
        # First convert value to int
        h = COLOR_MAPPING[int(value)]
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype='uint8')

    multiplier = 360 / n


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
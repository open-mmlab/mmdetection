# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Union

try:
    import seaborn as sns
except ImportError:
    sns = None
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.patches import Rectangle
from mmengine.utils import mkdir_or_exist


def imshow_mot_errors(*args, backend: str = 'cv2', **kwargs):
    """Show the wrong tracks on the input image.

    Args:
        backend (str, optional): Backend of visualization.
            Defaults to 'cv2'.
    """
    if backend == 'cv2':
        return _cv2_show_wrong_tracks(*args, **kwargs)
    elif backend == 'plt':
        return _plt_show_wrong_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()


def _cv2_show_wrong_tracks(img: Union[str, np.ndarray],
                           bboxes: np.ndarray,
                           ids: np.ndarray,
                           error_types: np.ndarray,
                           thickness: int = 2,
                           font_scale: float = 0.4,
                           text_width: int = 10,
                           text_height: int = 15,
                           show: bool = False,
                           wait_time: int = 100,
                           out_file: str = None) -> np.ndarray:
    """Show the wrong tracks with opencv.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (int, optional): Thickness of lines.
            Defaults to 2.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 0.4.
        text_width (int, optional): Width to draw id and score.
            Defaults to 10.
        text_height (int, optional): Height to draw id and score.
            Defaults to 15.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Visualized image.
    """
    if sns is None:
        raise ImportError('please run pip install seaborn')
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 5, but its {bboxes.shape[1]}.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]
    bbox_colors = [[int(255 * _c) for _c in bbox_color][::-1]
                   for bbox_color in bbox_colors]

    if isinstance(img, str):
        img = mmcv.imread(img)
    else:
        assert img.ndim == 3

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    for bbox, error_type, id in zip(bboxes, error_types, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = bbox_colors[error_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # FN does not have id and score
        if error_type == 1:
            continue

        # score
        text = '{:.02f}'.format(score)
        width = (len(text) - 1) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            text, (x1, y1 + text_height - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + text_height * 2,
            x1:x1 + width, :] = bbox_color
        cv2.putText(
            img,
            str(id), (x1, y1 + text_height * 2 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def _plt_show_wrong_tracks(img: Union[str, np.ndarray],
                           bboxes: np.ndarray,
                           ids: np.ndarray,
                           error_types: np.ndarray,
                           thickness: float = 0.1,
                           font_scale: float = 3.0,
                           text_width: int = 8,
                           text_height: int = 13,
                           show: bool = False,
                           wait_time: int = 100,
                           out_file: str = None) -> np.ndarray:
    """Show the wrong tracks with matplotlib.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (float, optional): Thickness of lines.
            Defaults to 0.1.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 3.0.
        text_width (int, optional): Width to draw id and score.
            Defaults to 8.
        text_height (int, optional): Height to draw id and score.
            Defaults to 13.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Original image.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 5, but its {bboxes.shape[1]}.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]

    if isinstance(img, str):
        img = plt.imread(img)
    else:
        assert img.ndim == 3
        img = mmcv.bgr2rgb(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(
        top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.rcParams['figure.figsize'] = img_shape[1], img_shape[0]

    for bbox, error_type, id in zip(bboxes, error_types, ids):
        x1, y1, x2, y2, score = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        left_top = (int(x1), int(y1))

        # bbox
        plt.gca().add_patch(
            Rectangle(
                left_top,
                w,
                h,
                thickness,
                edgecolor=bbox_colors[error_type],
                facecolor='none'))

        # FN does not have id and score
        if error_type == 1:
            continue

        # score
        text = '{:.02f}'.format(score)
        width = len(text) * text_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1]),
                      width,
                      text_height,
                      thickness,
                      edgecolor=bbox_colors[error_type],
                      facecolor=bbox_colors[error_type]))

        plt.text(
            left_top[0],
            left_top[1] + text_height + 2,
            text,
            fontsize=font_scale)

        # id
        text = str(id)
        width = len(text) * text_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1] + text_height + 1),
                      width,
                      text_height,
                      thickness,
                      edgecolor=bbox_colors[error_type],
                      facecolor=bbox_colors[error_type]))
        plt.text(
            left_top[0],
            left_top[1] + 2 * (text_height + 1),
            text,
            fontsize=font_scale)

    if out_file is not None:
        mkdir_or_exist(osp.abspath(osp.dirname(out_file)))
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if show:
        plt.draw()
        plt.pause(wait_time / 1000.)

    plt.clf()
    return img

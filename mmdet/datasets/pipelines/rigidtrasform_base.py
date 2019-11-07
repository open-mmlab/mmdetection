import mmcv
import numpy as np
from numpy import random
import cv2
from albumentations import (HorizontalFlip, VerticalFlip, Resize, CenterCrop,
                            ShiftScaleRotate, Transpose, RandomRotate90,
                            RandomCrop, Crop, Compose, RGBShift, GaussNoise,
                            RandomBrightness, RandomContrast, RandomGamma,
                            CLAHE, JpegCompression, HueSaturationValue,
                            RandomSizedCrop,Rotate)

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 255)


def visualize_bbox(img,
                   bbox,
                   class_id,
                   class_idx_to_name,
                   color=BOX_COLOR,
                   thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color,
                  thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img,
                class_name, (x_min, y_min - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT_COLOR,
                lineType=cv2.LINE_AA)
    return img


def get_aug(p=0.5, min_area=0., min_visibility=0.):
    return Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Rotate(p=0.5),
        ],
        bbox_params={
            'format': 'pascal_voc',
            'min_area': min_area,
            'min_visibility': min_visibility,
            'label_fields': ['category_id']
        },
        p=p)


def color_aug(p=0.5):
    return Compose(
        [
            RandomBrightness(p=0.5),
            RandomGamma(p=0.5),
            RGBShift(p=0.5),
        ],
        p=p)

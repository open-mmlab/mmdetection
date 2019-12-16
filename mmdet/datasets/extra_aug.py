import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from albumentations import (HorizontalFlip, VerticalFlip, Resize, CenterCrop,
                            ShiftScaleRotate, Transpose, RandomRotate90,
                            RandomCrop, Crop, Compose, RGBShift, GaussNoise,
                            RandomBrightness, RandomContrast, RandomGamma,
                            CLAHE, JpegCompression, HueSaturationValue, RandomSizedCrop)
import cv2

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
COUNT= 0

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
            VerticalFlip(),
            RandomRotate90(),
        ],
        bbox_params={
            'format': 'pascal_voc',
            'min_area': min_area,
            'min_visibility': min_visibility,
            'label_fields': ['category_id']
        },
        p=p)


class RigidTrasform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes, labels):
        DEGUG = True
        if DEGUG:
            global COUNT
            image_before = img.copy()
            for idx, bbox in enumerate(boxes):
                image_before = visualize_bbox(image_before, bbox, labels[idx], {1: 'positive'})
            mmcv.imwrite(image_before, "/home/wanglichao/AnalyticalToolkit/mmdetection/output/img_" + str(COUNT + 1) + "_before.jpg")
        
        annotations = {'image': img, 'bboxes': boxes, 'category_id': labels}
        aug = get_aug(p=self.p)
        annotations_aug = aug(**annotations)

        if DEGUG:
            print("\nafter", "*" * 50)
            image_after = annotations_aug['image'].copy()
            for idx, bbox in enumerate(annotations_aug['bboxes']):
                image_after = visualize_bbox(image_after, bbox, labels[idx], {1: 'positive'})
            mmcv.imwrite(image_after, "/home/wanglichao/AnalyticalToolkit/mmdetection/output/img_" + str(COUNT + 1) + "_after.jpg")
            COUNT = COUNT + 1
            if COUNT > 20:
                exit(1)

        return annotations_aug['image'].copy(), annotations_aug['bboxes'], annotations_aug['category_id']


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 rigid_trasform=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if rigid_trasform is not None:
            self.transforms.append(RigidTrasform(**rigid_trasform))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        ##debug
        DEGUG = False
        if DEGUG:
            global COUNT
            image_before = img.copy()
            for idx, bbox in enumerate(boxes):
                image_before = visualize_bbox(image_before, bbox, labels[idx], {1: 'positive'})
            mmcv.imwrite(image_before, "/home/wanglichao/mmdetection/output/img_" + str(COUNT + 1) + "_before.jpg")

        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        
        ##debug
        boxes = np.array(boxes).astype(np.float32)
        labels = np.array(labels).astype(np.int64)

        if len(boxes.shape) == 1:
            print("@" * 100)
            boxes = np.expand_dims(boxes, axis=0)

        if DEGUG:
            print("\nafter", "*" * 50)
            image_after = img.copy()
            for idx, bbox in enumerate(boxes):
                image_after = visualize_bbox(image_after, bbox, labels[idx], {1: 'positive'})
            mmcv.imwrite(image_after, "/home/wanglichao/mmdetection/output/img_" + str(COUNT + 1) + "_after.jpg")
            COUNT = COUNT + 1
            if COUNT > 20:
                exit(1)
        return img, boxes, labels

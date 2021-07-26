import numpy as np
import random
from numpy import random as numpy_random
import cv2
from mmdet.datasets import CocoDataset
import os
import torch
import math
import copy
from mmdet.datasets.pipelines import RandomCrop, LoadImageFromFile, LoadAnnotations
class Mosaic(object):
    """Mosaic augmentation.
    Given 4 images, Mosaic augmentation randomly crop a patch on each image
    and combine them into one output image. The output image is composed of
    the parts from each sub-image.
                        output image
                                cut_x
               +-----------------------------+
               |     image 0      | image 1  |
               |                  |          |
        cut_y  |------------------+----------|
               |     image 2      | image 3  |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               |                  |          |
               +-----------------------------+
    Args:
        size (tuple[int]): output image size in (h,w).
        min_offset (float | tuple[float]): Volume of the offset
            of the cropping window. If float, both height and width are
        dataset (torch.nn.Dataset): Dataset with augmentation pipeline.
    """

    def __init__(self, size=(640, 640), min_offset=0.2, dataset=None):

        assert isinstance(size, tuple)
        assert isinstance(size[0], int) and isinstance(size[1], int)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError('image size must > 0 in train mode')

        if isinstance(min_offset, float):
            assert 0 <= min_offset <= 1
            self.min_offset = (min_offset, min_offset)
        elif isinstance(min_offset, tuple):
            assert isinstance(min_offset[0], float) \
                   and isinstance(min_offset[1], float)
            assert 0 <= min_offset[0] <= 1 and 0 <= min_offset[1] <= 1
            self.min_offset = min_offset
        else:
            raise TypeError('Unsupported type for min_offset, '
                            'should be either float or tuple[float]')

        self.size = size
        self.dataset = dataset
        self.cropper = RandomCrop(crop_size=size, allow_negative_crop=True)
        self.num_sample = len(dataset)


    def __call__(self, results):
        """Call the function to mix 4 images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images and bounding boxes cropped.
        """
        # Generate the Mosaic coordinate (0.2 size -- 0.8 size)
        cut_y = numpy_random.randint(
            int(self.size[0] * self.min_offset[0]),
            int(self.size[0] * (1 - self.min_offset[0])))
        cut_x = numpy_random.randint(
            int(self.size[1] * self.min_offset[1]),
            int(self.size[1] * (1 - self.min_offset[1])))

        cut_position = (cut_y, cut_x)
        tmp_result = copy.deepcopy(results)
        # create the image buffer and mask buffer
        tmp_result['img'] = np.zeros(
            (self.size[0], self.size[1], *tmp_result['img'].shape[2:]),
            dtype=tmp_result['img'].dtype)
        for key in tmp_result.get('seg_fields', []):
            tmp_result[key] = np.zeros(
                (self.size[0], self.size[1], *tmp_result[key].shape[2:]),
                dtype=tmp_result[key].dtype)
        tmp_result['img_shape'] = self.size

        out_bboxes = []
        out_labels = []
        out_ignores = []

        for loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right'):
            if loc == 'top_left':
                # use the current image, top_left img is current img
                results_i = copy.deepcopy(results)
            else:
                # randomly sample a new image from the dataset
                index = numpy_random.randint(self.num_sample)
                results_i = copy.deepcopy(self.dataset.__getitem__(index))


            # compute the crop parameters
            crop_size, img_slices, paste_position = self._mosiac_combine(
                loc, cut_position)

            h, w, _ = results_i['img'].shape
            min_image_edge = min(h ,w)
            max_input_edge = max(self.size[0], self.size[1])
            scale = max(1, max_input_edge / min_image_edge)
            results_i['img'] = cv2.resize(results_i['img'], (int(scale * w), int(scale * h)), cv2.INTER_LINEAR)
            results_i['gt_bboxes'] = results_i['gt_bboxes']*scale

            bboxes = results_i['gt_bboxes']
            img = results_i['img']

            for box in bboxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cv2.imwrite("vis_ori_image.jpg", img)
            # randomly crop the image and segmentation mask
            self.cropper.crop_size = crop_size

            results_i = self.cropper(results_i)

            tmp_result['img'][img_slices] = results_i['img'].copy()

            for key in tmp_result.get('seg_fields', []):
                tmp_result[key][img_slices] = results_i[key].copy()

            results_i = self._adjust_coordinate(results_i, paste_position)

            out_bboxes.append(results_i['gt_bboxes'])
            out_labels.append(results_i['gt_labels'])
            out_ignores.append(results_i['gt_bboxes_ignore'])

            # bboxes = results_i['gt_bboxes']
            # img = results_i['img']
            bboxes = np.concatenate(out_bboxes, axis=0)
            img = tmp_result['img']
            for box in bboxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            cv2.imwrite("vis_mosaic.jpg", img)

        out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)
        out_ignores = np.concatenate(out_ignores, axis=0)

        tmp_result['gt_bboxes'] = out_bboxes
        tmp_result['gt_labels'] = out_labels
        tmp_result['gt_bboxes_ignore'] = out_ignores

        return tmp_result

    def _mosiac_combine(self, loc, cut_position):
        """Crop the subimage, change the label and mix the image.
        Args:
            loc (str): Index for the subimage, loc in ('top_left',
                'top_right', 'bottom_left', 'bottom_right').
            results (dict): Result dict from loading pipeline.
            img (numpy array): buffer for mosiac image, (H x W x 3).
            cut_position (tuple[int]): mixing center for 4 images, (y, x).
        Returns:
            bboxes: Result dict with images and bounding boxes cropped.
        """
        if loc == 'top_left':
            # Image 0: top left
            crop_size = cut_position
            img_slices = (slice(0, cut_position[0]), slice(0, cut_position[1]))
            paste_position = (0, 0)
        elif loc == 'top_right':
            # Image 1: top right
            crop_size = (cut_position[0], self.size[1] - cut_position[1])
            img_slices = (slice(0, cut_position[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = (0, cut_position[1])
        elif loc == 'bottom_left':
            # Image 2: bottom left
            crop_size = (self.size[0] - cut_position[0], cut_position[1])
            img_slices = (slice(cut_position[0],
                                self.size[0]), slice(0, cut_position[1]))
            paste_position = (cut_position[0], 0)
        elif loc == 'bottom_right':
            # Image 3: bottom right
            crop_size = (self.size[0] - cut_position[0],
                         self.size[1] - cut_position[1])
            img_slices = (slice(cut_position[0], self.size[0]),
                          slice(cut_position[1], self.size[1]))
            paste_position = cut_position

        return crop_size, img_slices, paste_position

    def _adjust_coordinate(self, results, paste_position):
        """Convert subimage coordinate to mosaic image coordinate.
         Args:
            results (dict): Result dict from :obj:`dataset`.
            paste_position (tuple[int]): paste up-left corner
                coordinate (y, x) in mosaic image.
        Returns:
            results (dict): Result dict with corrected bbox
                and mask coordinate.
        """

        for key in results.get('bbox_fields', []):
            box = results[key]
            box[:, 0::2] += paste_position[1]
            box[:, 1::2] += paste_position[0]
            results[key] = box
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size={self.size}, '
        repr_str += f'min_offset={self.min_offset})'
        return repr_str

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def random_perspective(
    img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes bboxes四个坐标点的x值 ，y值。
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets

class MosaicDetection(object):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        self._dataset = dataset
        self.input_dim = img_size
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels = []
            input_dim = self.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img_info = self._dataset.data_infos[index]
                ann_info = self._dataset.get_ann_info(index)
                img = cv2.imread(os.path.join(self._dataset.img_prefix, img_info["filename"]))
                _labels = np.concatenate([ann_info["bboxes"], ann_info["labels"][:, None]], axis=1)
                # img, _labels, _, _ = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            result = dict(gt_bboxes=mosaic_labels[:, :4], gt_labels=mosaic_labels[:, 4], img=mosaic_img)
            return result


if __name__ == "__main__":
    data_root = "/usr/videodate/dataset/subsetcoco/"
    annotations = "annotations/instances_train2017.json"
    dataset_yolox = CocoDataset(ann_file=data_root + annotations,
                                pipeline=[],
                                img_prefix=data_root + "train2017")
    mosaic_dataset = MosaicDetection(dataset=dataset_yolox, img_size=(640, 640), mosaic=True)
    for index in range(len(mosaic_dataset)):
        result = mosaic_dataset[index]
        img = result["img"]
        gt_bboxes = result["gt_bboxes"]
        for box in gt_bboxes:
            x1, y1 = box[:2].astype(np.int)
            x2, y2 = box[2:].astype(np.int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(f"../image/mosaic_yolox/{index}.jpg", img)

    # mmdet_sub_dataset = CocoDataset(ann_file=data_root + annotations,
    #                                 pipeline=[LoadImageFromFile(),
    #                                           LoadAnnotations()],
    #                                 img_prefix=data_root + "train2017")
    #
    # dataset_mmdet = CocoDataset(ann_file=data_root + annotations,
    #                             pipeline=[LoadImageFromFile(),
    #                                       LoadAnnotations(),
    #                                       Mosaic(dataset=mmdet_sub_dataset)],
    #                             img_prefix=data_root + "train2017")
    #
    # for index in range(len(dataset_mmdet)):
    #     result = dataset_mmdet[index]
    #     img = result["img"]
    #     gt_bboxes = result["gt_bboxes"]
    #     for box in gt_bboxes:
    #         x1, y1 = box[:2].astype(np.int)
    #         x2, y2 = box[2:].astype(np.int)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #     cv2.imwrite(f"../image/mosaic_mmdet/{index}.jpg", img)




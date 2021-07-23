from .coco import CocoDataset
import numpy as np
from functools import wraps
import os
import cv2
import random
import math
from mmcv import build_from_cfg
from .builder import DATASETS


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


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

        # create new boxes
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


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    image = padded_img

    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std
    # image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None):
        self.means = rgb_means
        self.std = std
        self.p = p

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if targets.shape[1] > 5:
            mixup = True
            ratios = targets[:, -1].copy()
            ratios_o = targets[:, -1].copy()
        else:
            mixup = False
            ratios = None
            ratios_o = None
        lshape = 6 if mixup else 5
        if len(boxes) == 0:
            targets = np.zeros((0, lshape), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        b_x_o = (boxes_o[:, 2] + boxes_o[:, 0]) * 0.5
        b_y_o = (boxes_o[:, 3] + boxes_o[:, 1]) * 0.5
        b_w_o = (boxes_o[:, 2] - boxes_o[:, 0]) * 1.0
        b_h_o = (boxes_o[:, 3] - boxes_o[:, 1]) * 1.0
        boxes_o[:, 0] = b_x_o
        boxes_o[:, 1] = b_y_o
        boxes_o[:, 2] = b_w_o
        boxes_o[:, 3] = b_h_o

        image_t = _distort(image)
        image_t, boxes = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        boxes = boxes.copy()
        # boxes [xyxy] 2 [cx,cy,w,h]
        b_x = (boxes[:, 2] + boxes[:, 0]) * 0.5
        b_y = (boxes[:, 3] + boxes[:, 1]) * 0.5
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.0
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.0
        boxes[:, 0] = b_x
        boxes[:, 1] = b_y
        boxes[:, 2] = b_w
        boxes[:, 3] = b_h

        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        if mixup:
            ratios_t = ratios[mask_b].copy()

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ratios_t = ratios_o

        labels_t = np.expand_dims(labels_t, 1)
        if mixup:
            ratios_t = np.expand_dims(ratios_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, ratios_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, targets_t


@DATASETS.register_module()
class COCODataset(CocoDataset):
    """
    COCO dataset class.
    """

    def __init__(self, *args, **kwargs):
        super(COCODataset, self).__init__(*args, **kwargs)
        self.preproc = TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                      std=(0.229, 0.224, 0.225))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])

        self.input_dim = (640, 640)

    # def __len__(self):
    #     return len(self.ids)

    def load_anno(self, index):
        id_ = self.ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        # load labels
        valid_objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        return res

    def pull_item(self, index):
        id_ = self.ids[index]

        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        # load image and preprocess
        img_file = os.path.join(self.img_prefix, "{:012}".format(id_) + ".jpg")

        img = cv2.imread(img_file)
        assert img is not None

        # load anno
        res = self.load_anno(index)
        img_info = (height, width)

        return img, res, img_info, id_

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, res, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, res, self.input_dim)

        return img, target, img_info, img_id


@DATASETS.register_module()
class MosaicDetection(CocoDataset):
    """Detection dataset wrapper that performs mixup for normal dataset.

    Parameters
    ----------
    dataset : Pytorch Dataset
        Gluon dataset object.
    *args : list
        Additional arguments for mixup random sampler.
    """

    def __init__(self,
                 dataset,
                 ann_file,
                 pipeline,
                 img_prefix=None,
                 mixup_scale=(0.5, 1.5),
                 scale=(0.1, 2),
                 enable_mixup=True,
                 *args,
                 **kwargs):
        dataset["pipeline"] = []
        dataset = build_from_cfg(dataset, DATASETS)

        self._dataset = dataset
        super(MosaicDetection, self).__init__(ann_file, pipeline, img_prefix=img_prefix, filter_empty_gt=False)

        self.preproc = TrainTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = scale
        self.shear = 2.0
        self.perspective = 0.0
        self.mixup_scale = mixup_scale
        self.mosaic = True
        self.enable_mixup = enable_mixup
        self.input_dim = (640, 640)
        self.input_size = self.input_dim

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img, gt, _, _ = self._train(idx)
        gt_bboxes = gt[:, 1:]
        gt_labels = gt[:, 0]
        w = gt_bboxes[:, 2].copy()
        h = gt_bboxes[:, 3].copy()
        cx = gt_bboxes[:, 0].copy()
        cy = gt_bboxes[:, 1].copy()
        gt_bboxes[:, 0] = cx - w / 2
        gt_bboxes[:, 1] = cy - h / 2
        gt_bboxes[:, 2] = cx + w / 2
        gt_bboxes[:, 3] = cy + h / 2
        results = {'img': img, 'gt_bboxes': gt_bboxes,
                   'gt_labels': gt_labels}
        results = self.pipeline(results)

        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])

        # show_img = img.copy() * std + mean
        # show_img = show_img * 255

        # for i in range(len(gt_bboxes)):
        #     box = gt_bboxes[i]
        #     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #     cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.imwrite(f"img/{idx}.jpg", show_img)

        return results

    def _train(self, idx):
        if self.mosaic:
            labels4 = []
            # Never change
            input_dim = self.input_dim
            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_dim[0], 1.5 * input_dim[0]))
            xc = int(random.uniform(0.5 * input_dim[1], 1.5 * input_dim[1]))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i, index in enumerate(indices):
                img, _labels, _, _ = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_dim[0] / h0, 1. * input_dim[1] / w0)
                interp = cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=interp)
                (h, w) = img.shape[:2]

                if i == 0:  # top left
                    # base image with 4 tiles
                    img4 = np.full(
                        (input_dim[0] * 2, input_dim[1] * 2, img.shape[2]), 114, dtype=np.uint8
                    )
                    # xmin, ymin, xmax, ymax (large image)
                    x1a, y1a, x2a, y2a = (max(xc - w, 0), max(yc - h, 0), xc, yc,)
                    # xmin, ymin, xmax, ymax (small image)
                    x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h,)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, input_dim[1] * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(input_dim[0] * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, input_dim[1] * 2), min(input_dim[0] * 2, yc + h)  # noqa
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                padw = x1a - x1b
                padh = y1a - y1b

                labels = _labels.copy()  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
                if _labels.size > 0:  # Normalized xywh to pixel xyxy format
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                labels4.append(labels)

            if len(labels4):
                labels4 = np.concatenate(labels4, 0)
                np.clip(labels4[:, 0], 0, 2 * input_dim[1], out=labels4[:, 0])
                np.clip(labels4[:, 1], 0, 2 * input_dim[0], out=labels4[:, 1])
                np.clip(labels4[:, 2], 0, 2 * input_dim[1], out=labels4[:, 2])
                np.clip(labels4[:, 3], 0, 2 * input_dim[0], out=labels4[:, 3])

            img4, labels4 = random_perspective(
                img4,
                labels4,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_dim[0] // 2, -input_dim[1] // 2],
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if self.enable_mixup and not len(labels4) == 0:
                # TODO: Test
                img4, labels4 = self.mixup(img4, labels4, self.input_size)
            # change size
            mix_img, padded_labels = self.preproc(img4, labels4, self.input_size)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, img_info, int(idx)
        else:
            img, label, img_info, idx = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_size)
            return img, label, img_info, int(idx)

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4], cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5]
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        # origin_img = origin_img.transpose(())
        return origin_img.astype(np.uint8), origin_labels

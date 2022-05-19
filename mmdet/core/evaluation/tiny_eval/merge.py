import numpy as np
from copy import deepcopy
import json
import os
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import copy
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
import math

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


def IOD(dets):
    def insect_boxes(box1, boxes):
        sx1, sy1, sx2, sy2 = box1[:4]
        tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        ix1 = np.where(tx1 > sx1, tx1, sx1)
        iy1 = np.where(ty1 > sy1, ty1, sy1)
        ix2 = np.where(tx2 < sx2, tx2, sx2)
        iy2 = np.where(ty2 < sy2, ty2, sy2)
        return np.array([ix1, iy1, ix2, iy2]).transpose((1, 0))

    def bbox_area(boxes):
        s = np.zeros(shape=(boxes.shape[0],), dtype=np.float32)
        tx1, ty1, tx2, ty2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        h = (tx2 - tx1)
        w = (ty2 - ty1)
        valid = np.all(np.array([h > 0, w > 0]), axis=0)
        s[valid] = (h * w)[valid]
        return s

    def bbox_iod(dets, gts, eps=1e-12):
        iods = np.zeros(shape=(dets.shape[0], gts.shape[0]), dtype=np.float32)
        dareas = bbox_area(dets)
        for i, (darea, det) in enumerate(zip(dareas, dets)):
            idet = insect_boxes(det, gts)
            iarea = bbox_area(idet)
            iods[i, :] = iarea / (darea + eps)
        return iods

    def xywh2xyxy(boxes):
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes

    from copy import deepcopy
    return bbox_iod(xywh2xyxy(deepcopy(dets)), xywh2xyxy(deepcopy(dets)))


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # ##################################################3 add by hui ######################################3
    def translate(self, delta_x, delta_y, clip=True):
        bbox = copy.deepcopy(self)  # BoxList(self.bbox, self.size, mode=self.mode)
        bbox.bbox[:, 0] += delta_x
        bbox.bbox[:, 1] += delta_y
        if self.mode == 'xyxy':
            bbox.bbox[:, 2] += delta_x
            bbox.bbox[:, 3] += delta_y
        if clip:
            bbox = bbox.clip_to_image()
        return bbox

    ######################################3#################################################################

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


def torch_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    keep = []
    if dets.numel() == 0:
        return torch.LongTensor(keep)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
    _, order = scores.sort(0, descending=True)

    while order.numel() > 0:
        # i = order[0]
        # keep.append(i)
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = (ovr <= thresh).nonzero().squeeze()
        if inds.numel() == 0:
            break
        order = order[inds + 1]

    return torch.LongTensor(keep)


class MergeResult(object):
    def __init__(self, use_nms=True, wbf=False, bwbf=False, use_iod=False, nms_th=0.5, score_thr=0.05):
        self.use_nms = use_nms
        self.use_iod = use_iod
        self.nms_th = nms_th
        self.wbf = wbf
        self.bwbf = bwbf
        self.score_thr = score_thr

    def merge_result(self, corners, results, scores=None):
        merge_result = self.translate_bboxes(corners, results)
        no_empty_result = sum([len(result) > 0 for result in results])
        if no_empty_result > 1 and self.use_nms:  # only when no_empty sub result > 1, need nms to merge in overlap area
            merge_result, keep = self.nms(merge_result, scores)
        return merge_result

    def merge_maskrcnn_benchmark_result(self, corners, results, im_scales=None, image_size=None):
        import torch
        # from maskrcnn_benchmark.structures.boxlist_ops import BoxList
        # from .bounding_box import BoxList

        def result_fmt(result):
            bbox = result.bbox
            labels = result.extra_fields["labels"].reshape(-1, 1).float()
            scores = result.extra_fields["scores"].reshape(-1, 1)
            det_result = torch.cat([bbox, labels, scores], dim=1).detach().cpu().numpy()
            return det_result

        input_BoxList = isinstance(results[0], BoxList)

        if input_BoxList:
            assert im_scales is not None and image_size is not None, ''

        if input_BoxList:
            det_results = []
            for result, im_scale in zip(results, im_scales):
                det_result = result_fmt(result)
                det_result[:, :4] = det_result[:, :4] / np.array([im_scale[1], im_scale[0], im_scale[1], im_scale[0]])
                det_results.append(det_result)
        else:
            det_results = results
        det_results = self.translate_bboxes(corners, det_results)
        if len(det_results) == 0: return []
        _, keep = self.nms(det_results[:, :4], det_results[:, 5])
        det_results = det_results[keep]

        if input_BoxList:
            merge_result = BoxList(torch.Tensor(det_results[:, :4]), image_size, 'xyxy')
            merge_result.add_field("labels", torch.Tensor(det_results[:, 4]))
            merge_result.add_field("scores", torch.Tensor(det_results[:, 5]))
        else:
            merge_result = det_results
        return merge_result

    def translate_bboxes(self, corners, results):
        """
        :param corners: corner of all sub image
        :param results: result of all sub image, results[i] = np.array([[x1, y1, x2, y2,...]...])
        :return:
        """
        merge_result = []
        for corner, result in zip(corners, results):
            if len(result) == 0:
                continue
            result = result.copy()
            result[:, [0, 2]] += corner[0]
            result[:, [1, 3]] += corner[1]
            merge_result.extend(result.tolist())
        merge_result = np.array(merge_result)
        return merge_result

    def BoundingBox(self, bbox, x_border, y_border):
        keep = []
        for i in range(len(bbox)):
            for x_b in range(len(x_border)):
                if x_border[x_b][0] < bbox[i][0] < x_border[x_b][1] or \
                        x_border[x_b][0] < bbox[i][2] < x_border[x_b][1]:
                    keep.append(i)
                    continue
            for y_b in range(len(y_border)):
                if y_border[y_b][0] < bbox[i][1] < y_border[y_b][1] or \
                        y_border[y_b][0] < bbox[i][3] < y_border[y_b][1]:
                    keep.append(i)
        return list(set(keep))

    def get_scale_factor(self, json_file):
        shape_dict = dict()
        for i in range(len(json_file["old_images"])):
            shape_dict[json_file["old_images"][i]['id']] = [json_file["old_images"][i]['width'],
                                                            json_file["old_images"][i]['height'],
                                                            json_file["old_images"][i]['width'],
                                                            json_file["old_images"][i]['height']
                                                            ]
        return shape_dict

    def nms(self, merge_result, scores):

        # from maskrcnn_benchmark.layers import nms as _box_nms
        import torch
        if scores is None:
            scores = torch.ones(size=(len(merge_result),))
        if not isinstance(scores, torch.Tensor):
            scores = torch.Tensor(scores)
        merge_result = torch.Tensor(merge_result)

        keep = torch_nms(merge_result, scores, self.nms_th)
        merge_result = merge_result[keep].detach().cpu().numpy()
        return merge_result, keep.detach().cpu().numpy()


def xywh2xyxy(boxes):
    x, y, w, h = boxes.T
    x2, y2 = x + w, y + h
    return np.array([x, y, x2, y2]).T


def xyxy2xywh(boxes):
    x1, y1, x2, y2 = boxes.T
    return np.array([x1, y1, x2 - x1, y2 - y1]).T


class COCOMergeResult(MergeResult):

    def __turn_det_result(self, bbox, image_id, old_det_result):
        det_result = deepcopy(old_det_result)
        det_result['bbox'] = xyxy2xywh(np.array(bbox)).tolist()
        det_result['image_id'] = image_id
        return det_result

    def __load_coco_data_map(self, corner_gts):

        merge_image_ids = [image_info['id'] for image_info in corner_gts['old_images']]
        image_id_to_image_info = {image_info['id']: image_info for image_info in corner_gts['images']}

        filename_to_merge_image_id = {image_info['file_name']: image_info['id'] for image_info in
                                      corner_gts['old_images']}
        merge_image_id_to_image_ids = {}
        for image_info in corner_gts['images']:
            merge_image_id = filename_to_merge_image_id[image_info['file_name']]
            if merge_image_id not in merge_image_id_to_image_ids:
                merge_image_id_to_image_ids[merge_image_id] = [image_info['id']]
            else:
                merge_image_id_to_image_ids[merge_image_id].append(image_info['id'])
        assert len(merge_image_ids) == len(merge_image_id_to_image_ids)
        return merge_image_ids, image_id_to_image_info, merge_image_id_to_image_ids

    def __load_det_data_map(self, det_data):
        image_id_to_det_boxes = {}
        for det_bbox in det_data:
            if det_bbox['image_id'] not in image_id_to_det_boxes:
                image_id_to_det_boxes[det_bbox['image_id']] = [det_bbox]
            else:
                image_id_to_det_boxes[det_bbox['image_id']].append(det_bbox)
        return image_id_to_det_boxes

    def __save_file(self, json_data, src_path, dst_path):
        if os.path.isdir(dst_path):
            f_dir, f_name = os.path.split(src_path)
            f, ext = os.path.splitext(f_name)
            save_pth = os.path.join(dst_path,
                                    '{}_merge_nms{}{}'.format(f, self.nms_th if self.use_nms else 'None', ext))
        else:
            save_pth = dst_path
        json.dump(json_data, open(save_pth, 'w'))
        print('[COCOMergeResult]: save file to', os.path.abspath(save_pth))
        return save_pth

    def __call__(self, corner_gt_file_path, src_det_file_path, dst_det_file_path=None):
        det_data = json.load(open(src_det_file_path))
        corner_gts = json.load(
            open(corner_gt_file_path))
        Scale_factor_dict = self.get_scale_factor(corner_gts)
        # create data map
        merge_image_ids, image_id_to_image_info, merge_image_id_to_image_ids = self.__load_coco_data_map(corner_gts)
        image_id_to_det_boxes = {image_id: [] for image_id in image_id_to_image_info}
        image_id_to_det_boxes.update(self.__load_det_data_map(det_data))

        merge_image_id_to_det_results = {id: [] for id in merge_image_ids}
        merge_image_id_to_corners = {id: [] for id in merge_image_ids}
        for merge_image_id, image_ids in merge_image_id_to_image_ids.items():
            for image_id in image_ids:
                merge_image_id_to_det_results[merge_image_id].append(image_id_to_det_boxes[image_id])
                merge_image_id_to_corners[merge_image_id].append(image_id_to_image_info[image_id]['corner'])

        # merge all det result
        all_merge_det_results = []
        for merge_image_id in tqdm(merge_image_id_to_det_results):
            corners = merge_image_id_to_corners[merge_image_id]
            det_results_list = merge_image_id_to_det_results[merge_image_id]

            # merge all boxes to origin image
            old_det_results, det_bboxes = [], []
            for det_results in det_results_list:
                old_det_results.extend(det_results)
                det_bboxes.append(np.array([xywh2xyxy(np.array(det_result['bbox']))
                                            for det_result in det_results]))
            scale_factor = Scale_factor_dict[merge_image_id]

            if self.bwbf:
                width = corners[0][2]
                height = corners[0][3]
                x_stride = math.ceil(scale_factor[0] / (width - 30))
                y_stride = math.ceil(scale_factor[1] / (height - 30))
                x_border = []
                y_border = []
                x_s = 0
                y_s = 0
                for i in range(1, x_stride + 1):
                    x_e = x_s + width
                    x_s = x_e - 30
                    if x_e > scale_factor[0]:
                        x_e = x_e - width + 30
                        x_s = scale_factor[0] - width
                        x_border[-1] = [x_s, x_e]
                        continue
                    x_border.append([x_s, x_e])
                for i in range(1, y_stride + 1):
                    y_e = y_s + height
                    y_s = y_e - 30
                    if y_e > scale_factor[1]:
                        y_e = y_e - height + 30
                        y_s = scale_factor[1] - height
                        y_border[-1] = [y_s, y_e]
                        continue
                    y_border.append([y_s, y_e])

                merge_results = []
                merge_boxes = self.translate_bboxes(corners, det_bboxes)
                keep = self.BoundingBox(merge_boxes, x_border, y_border)
                ResBox = [merge_boxes[i] for i in range(len(merge_boxes)) if i not in keep]
                BBox = merge_boxes[keep]

                scores = [det_result['score'] for det_result in old_det_results]
                ResScore = [scores[i] for i in range(len(scores)) if i not in keep]
                Bscores = np.array(scores)[keep]
                if len(BBox) > 0:
                    BBox, Bscores, labels = weighted_boxes_fusion([BBox / scale_factor], [Bscores],
                                                                  [np.ones_like(Bscores)],
                                                                  weights=None, iou_thr=0.5)
                    BBox = BBox * scale_factor
                    for i in range(len(BBox)):
                        merge_results.append({"image_id": merge_image_id,
                                              'bbox': [float(BBox[i][0]), float(BBox[i][1]),
                                                       float(BBox[i][2] - BBox[i][0]),
                                                       float(BBox[i][3] - BBox[i][1])], 'score': float(Bscores[i]),
                                              'category_id': 1})
                if len(ResBox) > 0:
                    for i in range(len(ResBox)):
                        merge_results.append({"image_id": merge_image_id,
                                              'bbox': [float(ResBox[i][0]), float(ResBox[i][1]),
                                                       float(ResBox[i][2] - ResBox[i][0]),
                                                       float(ResBox[i][3] - ResBox[i][1])],
                                              'score': float(ResScore[i]), 'category_id': 1})

            elif self.use_nms:
                from functools import reduce
                merge_boxes = self.translate_bboxes(corners, det_bboxes)
                scores = [det_result['score'] for det_result in old_det_results]
                scores = np.array(scores)
                valid_idx = np.where(scores > self.score_thr)
                scores = scores[valid_idx]
                merge_boxes = merge_boxes[valid_idx]
                merge_boxes, keeps = self.nms(merge_boxes, scores)
                # old_det_results = [old_det_results[keep] for keep in keeps]
                scores = scores[keeps]
                merge_results = []
                for i in range(len(merge_boxes)):
                    merge_results.append({"image_id": merge_image_id,
                                          'bbox': [float(merge_boxes[i][0]), float(merge_boxes[i][1]),
                                                   float(merge_boxes[i][2] - merge_boxes[i][0]),
                                                   float(merge_boxes[i][3] - merge_boxes[i][1])],
                                          'score': float(scores[i]),
                                          'category_id': 1})

                # turn bbox to det_result
                # merge_results = []
                # for bbox, old_det_result in zip(merge_boxes, old_det_results):
                #     det_result = self.__turn_det_result(bbox, merge_image_id, old_det_result)
                #     merge_results.append(det_result)
            elif self.use_iod:
                merge_boxes = self.translate_bboxes(corners, det_bboxes)
                scores = [det_result['score'] for det_result in old_det_results]

                merge_boxes, keeps = self.nms(merge_boxes, scores)
                if len(merge_boxes) > 0:
                    temp_score = np.array(scores)[keeps]
                    temp_idx = []
                    iod = IOD(merge_boxes)
                    for i in range(len(iod)):
                        invalid_box = np.array(np.where(iod[i] > 0.98)).squeeze(axis=0)
                        invalid_box = [idx for idx in invalid_box if temp_score[idx] < temp_score[i]]
                        # invalid_box = np.where(temp_score[invalid_box] < temp_score[i])
                        temp_idx = np.union1d(invalid_box, temp_idx)
                    valid_idx = [i for i in range(len(merge_boxes)) if i not in temp_idx]

                    keeps = keeps[valid_idx]
                    merge_boxes = merge_boxes[valid_idx]
                old_det_results = [old_det_results[keep] for keep in keeps]

                # turn bbox to det_result
                merge_results = []
                for bbox, old_det_result in zip(merge_boxes, old_det_results):
                    det_result = self.__turn_det_result(bbox, merge_image_id, old_det_result)
                    merge_results.append(det_result)
            elif self.wbf:
                merge_results = []
                merge_boxes = self.translate_bboxes(corners, det_bboxes)
                scores = [det_result['score'] for det_result in old_det_results]
                if len(merge_boxes) > 0:
                    merge_boxes, scores, labels = weighted_boxes_fusion([merge_boxes / scale_factor], [scores],
                                                                        [np.ones_like(scores)],
                                                                        weights=None, iou_thr=0.5)

                    merge_boxes = merge_boxes * scale_factor
                    for i in range(len(merge_boxes)):
                        merge_results.append({"image_id": merge_image_id,
                                              'bbox': [float(merge_boxes[i][0]), float(merge_boxes[i][1]),
                                                       float(merge_boxes[i][2] - merge_boxes[i][0]),
                                                       float(merge_boxes[i][3] - merge_boxes[i][1])],
                                              'score': float(scores[i]),
                                              'category_id': 1})

                # turn bbox to det_result
            all_merge_det_results.extend(merge_results)

        save_pth = None
        if dst_det_file_path is not None:
            save_pth = self.__save_file(all_merge_det_results, src_det_file_path, dst_det_file_path)
        return all_merge_det_results, save_pth


def fusion(json1, json2, test_json, weights):
    inf_id = [n["id"] for n in test_json["images"]]
    fusion_result = []
    i = 0
    u = 0
    for idx, id in tqdm(enumerate(inf_id)):
        model1_bbox = []
        model1_score = []
        model1_label = []

        model2_bbox = []
        model2_score = []
        model2_label = []
        img_w = test_json["images"][idx]["width"]
        img_h = test_json["images"][idx]["height"]
        scale_factor = np.array([img_w, img_h, img_w, img_h])
        while i < len(json1) and json1[i]["image_id"] == id:

            temp_box = np.array([json1[i]['bbox'][0], json1[i]['bbox'][1], json1[i]['bbox'][2] + json1[i]['bbox'][0], \
                                 json1[i]['bbox'][3] + json1[i]['bbox'][1]])

            if (temp_box[2] - temp_box[0]) * (temp_box[3] - temp_box[1]) < 1600:
                model1_bbox.append(temp_box / scale_factor)
                model1_score.append(json1[i]['score'])
                model1_label.append(json1[i]['category_id'])
            i += 1
        while u < len(json2) and json2[u]["image_id"] == id:
            temp_box = np.array([json2[u]['bbox'][0], json2[u]['bbox'][1], json2[u]['bbox'][2] + json2[u]['bbox'][0], \
                                 json2[u]['bbox'][3] + json2[u]['bbox'][1]])
            if (temp_box[2] - temp_box[0]) * (temp_box[3] - temp_box[1]) < 1600:
                model2_bbox.append(temp_box / scale_factor)
                model2_score.append(json2[u]['score'])
                model2_label.append(json2[u]['category_id'])
            u += 1
        all_box = np.array([model1_bbox, model2_bbox])
        all_score = np.array([model1_score, model2_score])
        all_label = np.array([model1_label, model2_label])

        boxes, scores, labels = weighted_boxes_fusion(all_box, all_score, all_label, weights=weights, iou_thr=0.55,
                                                      skip_box_thr=0.0001, conf_type='avg')

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes = boxes * scale_factor

        for box in range(len(boxes)):
            fusion_result.append(
                {"image_id": id, "bbox": list(boxes[box]), "score": float(scores[box]),
                 "category_id": int(labels[box])})

    return fusion_result


if __name__ == '__main__':
    from mmdet.core.evaluation.cocoeval import COCOeval
    from pycocotools.coco import COCO
    import matplotlib.pyplot as plt

    # result_merge, json_result_file = COCOMergeResult(use_nms=True, bwbf=False, wbf=False, use_iod=False, nms_th=0.5,
    #                                                  score_thr=0.05)(
    #     r'/home/kris/PycharmProjects/ECCVW/mmdetection-master/data/dense/'
    #     r'annotation/Sub_training/sub_test/overlap30/tinyset_test_overlap30.json',
    #     '../../../../tools/result/temp/hrw40/result.bbox.json',
    #     '../../../../tools/result/temp/hrw40/merge_result.bbox.json'
    # )
    ann_file = r'/home/kris/PycharmProjects/ECCVW/mmdetection-master/data/dense/' \
               r'annotation/all_test/GT/tiny_set_test_single.json'
    #
    # cocoGt = COCO(ann_file)
    # cocoDt = cocoGt.loadRes(result_merge)
    # cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    result_merge1 = r"../../../../tools/result/merge_result.bbox.json"

    # result_merge2 = r"../../../../tools/result/merge_result.bbox.json"
    # result_merge3 = r"../../../../tools/result/temp/hrw40_decrease/merge_result.bbox.json"
    cocoGt1 = COCO(ann_file)
    cocoDt1 = cocoGt1.loadRes(result_merge1)
    cocoEval1 = COCOeval(cocoGt1, cocoDt1, 'bbox')
    cocoEval1.evaluate()
    cocoEval1.accumulate()
    cocoEval1.summarize()

    #cocoGt2 = COCO(ann_file)
    # cocoDt2 = cocoGt2.loadRes(result_merge2)
    # cocoEval2 = COCOeval(cocoGt2, cocoDt2, 'bbox')
    # cocoEval2.evaluate()
    # cocoEval2.accumulate()
    # cocoEval2.summarize()
    #
    # cocoGt3 = COCO(ann_file)
    # cocoDt3 = cocoGt3.loadRes(result_merge3)
    # cocoEval3 = COCOeval(cocoGt3, cocoDt3, 'bbox')
    # cocoEval3.evaluate()
    # cocoEval3.accumulate()
    # cocoEval3.summarize()

    #
    # pr_array2 = cocoEval1.eval['precision'][1, :, 0, 1, 0]
    # pr_array1 = cocoEval.eval['precision'][1, :, 0, 1, 0]
    # pr_array3 = cocoEval2.eval['precision'][1, :, 0, 1, 0]
    # pr_array4 = cocoEval3.eval['precision'][1, :, 0, 1, 0]
    # x = np.arange(0.0, 1.01, 0.01)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.xlim(0, 1.0)
    # plt.ylim(0, 1.01)
    # plt.grid(True)
    # plt.plot(x, pr_array1, label='base')
    # plt.plot(x, pr_array2, label='softmax')
    # plt.plot(x, pr_array3, label='sigmoid')
    # plt.plot(x, pr_array4, label='decrease')
    # plt.legend(loc="lower left")
    # plt.savefig('pr.jpg')
    # plt.show()

import cv2
import torch
from torch import nn
# from detectron2.utils.comm import get_world_size
# from detectron2.structures import pairwise_iou, Boxes
# from .data import CenterNetCrop
import torch.nn.functional as F
import numpy as np
# from detectron2.structures import Boxes, ImageList, Instances
import itertools
from typing import Any, Dict, List, Tuple, Union
import math
from enum import IntEnum, unique
from torch import device
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat

# __all__ = ['reduce_sum', '_transpose']

INF = 1000000000

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


if TORCH_VERSION < (1, 8):
    _maybe_jit_unused = torch.jit.unused
else:

    def _maybe_jit_unused(x):
        return x

def batched_nms(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        # fp16 does not have enough range for batched NMS
        return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep

def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    # if boxlist.has('pred_boxes'):
    #     boxes = boxlist.pred_boxes.tensor
    #     labels = boxlist.pred_classes
    # else:
    #     boxes = boxlist.proposal_boxes.tensor
    #     labels = boxlist.proposal_boxes.tensor.new_zeros(
    #         len(boxlist.proposal_boxes.tensor))
    boxes = boxlist[:,:4]
    labels = boxlist[:,-1]
    scores = boxlist[:,4]
    
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    @_maybe_jit_unused
    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    @_maybe_jit_unused
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix

    Args:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


def _transpose(training_targets, num_loc_list):
    '''
    This function is used to transpose image first training targets to 
        level first ones
    :return: level first training targets
    '''
    for im_i in range(len(training_targets)):
        training_targets[im_i] = torch.split(
            training_targets[im_i], num_loc_list, dim=0)

    targets_level_first = []
    for targets_per_level in zip(*training_targets):
        targets_level_first.append(
            torch.cat(targets_per_level, dim=0))
    return targets_level_first

'''
def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor
'''
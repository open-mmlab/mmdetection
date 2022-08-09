import cv2
import torch
import numpy as np
from torch.nn.modules.utils import _pair
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES
from .formating import to_tensor


@PIPELINES.register_module()
class BGR2RGB:
    """Convert normalized image back to int8
    """

    def __call__(self, results):
        """Call convert normalized image back to uint8

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Converted results, results['img'] has been converted to uint8.
        """
        img_bgr = results['img']
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results['img'] = img_rgb
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class IPUFormatBundle:
    """formatting bundle for IPU.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255),
                 pad_dic=dict()):
        self.img_to_float = img_to_float
        self.pad_val = pad_val
        self.pad_dic = pad_dic

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        for key, padding in self.pad_dic.items():
            dims = results[key].shape
            target_dim = padding.get('dim',0)
            pad_num = padding['num'] - dims[target_dim]
            assert pad_num >= 0, f"{key}: target padding num is {padding['num']}, but current dim is {dims[target_dim]}"
            pad_tuples = [(0, 0)]*len(dims)
            pad_tuples[target_dim] = (0, pad_num)
            results[key] = np.pad(results[key], pad_tuples, 'constant', constant_values=(0, 0))

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # float32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=True, pad_dims=0)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'


@PIPELINES.register_module()
class IPUCollect:
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg'),
                 meta_tensor_keys=tuple(),
                 meta_on=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_tensor_keys = meta_tensor_keys
        self.meta_on = meta_on
        self.shape_dic = {}

    def check_shape(self, name, shape):
        if name in self.shape_dic:
            org_shape = self.shape_dic[name]
            assert shape == org_shape, f'tensor({name}), org shape is {org_shape}, now is {shape}'
        else:
            self.shape_dic[name] = shape

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in self.meta_tensor_keys:
                img_meta[key] = torch.from_numpy(results[key])
            else:
                img_meta[key] = results[key]
        if self.meta_on:
            data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]#.data
            self.check_shape(key, results[key].data.shape)
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class GetTargetsOutsideForYolo:
    """Convert normalized image back to int8
    """
    def __init__(self, featmap_sizes, num_levels=3):
        s1, s2, s3 = featmap_sizes
        self.featmap_sizes = [torch.Size([s1, s1]), torch.Size([s2, s2]), torch.Size([s3, s3])]
        self.num_levels = num_levels
        self.model = None

    def __call__(self, results):
        """Call convert normalized image back to uint8

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Converted results, results['img'] has been converted to uint8.
        """
        img = results['img']
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        target_maps_list, neg_maps_list = self.get_targets_outside([gt_bboxes.data.clone()], [gt_labels.data.clone()])
        target_maps_list = [ele.squeeze(0) for ele in target_maps_list]
        neg_maps_list = [ele.squeeze(0) for ele in neg_maps_list]
        results['target_maps_list'] = target_maps_list
        results['neg_maps_list'] = neg_maps_list
        return results

    def get_targets_outside(self, gt_bboxes, gt_labels):
        mlvl_anchors = self.model.bbox_head.prior_generator.grid_priors(self.featmap_sizes, device='cpu')
        anchor_list = [mlvl_anchors for _ in range(1)]
        responsible_flag_list = []
        for img_id in range(1):
            responsible_flag_list.append(
                self.model.bbox_head.prior_generator.responsible_flags(self.featmap_sizes,
                                                    gt_bboxes[img_id], device='cpu'))
        target_maps_list, neg_maps_list = self.model.bbox_head.get_targets(
                anchor_list, responsible_flag_list, gt_bboxes, gt_labels)#[gt_bboxes.shape(96,4),], [gt_labels.shape(96),]
        return target_maps_list, neg_maps_list

    def set_model_in_ipu_mode(self, model):
        self.model = model

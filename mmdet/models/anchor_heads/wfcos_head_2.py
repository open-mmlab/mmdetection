"""Watershed-FCOS Head

A head that uses principles from both the Deep Watershed Detector paper as well
as the FCOS paper.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    January 06, 2020
"""
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob

INF = 1e8


class WFCOSHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 max_energy,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=None,
                 loss_bbox=None,
                 loss_energy=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 split_convs=False,
                 r=500.):
        """
        Creates a head based on FCOS that uses an energy_preds map, not centerness
        Args:
            num_classes (int): Number of classes to output.
            in_channels (int): Number of input channels.
            max_energy (int): Quantization of energy_preds. How much to split the
                energy_preds values by.
            feat_channels (int): Number of feature channels in each of the
                stacked convolutions.
            stacked_convs (int): Number of stacked convolutions to have.
            strides (tuple): Stride value for each of the heads.
            regress_ranges (tuple): The regression range for each of the heads.
            loss_cls (dict): A description of the loss to use for the
                classfication output.
            loss_bbox (dict): A description of the loss to use for the bbox
                output.
            loss_energy (dict): A description of the loss to use for the
                energy_preds map output.
            conv_cfg (dict): A description of the configuration of the
                convolutions in the stacked convolution.
            norm_cfg (dict): A description of the normalization configuration of
                the layers of the stacked convolution.
            split_convs (bool): Whether or not to split the classification and
                energy_preds map convolution stacks. False means that the
                classification energy_preds map shares the same convolution stack.
                Defaults to False.
            r (float): r variable in the energy map target equation.
        """
        super(WFCOSHead, self).__init__()

        # To avoid mutable default values
        if loss_cls is None:
            loss_cls = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)

        if loss_bbox is None:
            loss_bbox = dict(type='IoULoss', loss_weight=1.0)

        if loss_energy is None:
            loss_energy = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)

        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

        # Save the different arguments to self
        self.num_classes = num_classes  # We assign class 0 as the null class
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_energy = build_loss(loss_energy)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        # WFCOS variables
        self.max_energy = max_energy
        self.split_convs = split_convs
        self.r = r

        # Now create the layers
        self._init_layers()

    def _init_layers(self):
        """Initialize each of the layers needed."""
        self.cls_convs = nn.ModuleList()
        self.energy_convs = None if not self.split_convs else nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        # Create the stacked convolutions
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels

            # Make the different convolution stacks
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            if self.split_convs:
                self.energy_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None
                    )
                )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        # Classifier convolution
        self.wfcos_cls = nn.Conv2d(
            self.feat_channels, self.num_channels, 3, padding=1)
        # Bounding box regression convolution
        self.wfcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # Energy map convolution
        self.wfcos_energy = nn.Conv2d(self.feat_channels, self.max_energy,
                                      1, padding=0)

        # Scaling factor for the different heads
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize the weights for all the layers with a normal dist."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        if self.split_convs:
            for m in self.energy_convs:
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.wfcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.wfcos_reg, std=0.01)
        normal_init(self.wfcos_energy, std=0.01)

    def forward(self, feats):
        """Run forwards on the network.

        Args:
            feats (tuple): tuple of torch tensors handed off from the neck.
                Expects the use of FPN as the neck, giving multiple feature
                tensors.

        Returns:
            (tuple): A tuple of 3-tuples of tensors, each 3-tuple representing
                cls_score, bbox_pred, energy_preds of the different feature layers.
        """
        # Use a multi_apply function to run forwards on each feats tensor
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Runs forwards on a single feature level."""
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.wfcos_cls(cls_feat)

        if self.split_convs:
            energy_feat = x
            for energy_layer in self.energy_convs:
                energy_feat = energy_layer(energy_feat)
            energy = self.wfcos_energy(energy_feat)
        else:
            energy = self.wfcos_energy(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.wfcos_reg(reg_feat)).float().exp()

        return cls_score, bbox_pred, energy

    @force_fp32(apply_to=('label_preds', 'bbox_preds', 'energy_preds'))
    def loss(self,
             label_preds,
             bbox_preds,
             energy_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        """Calculates loss for each of the head outputs.

        Calculates a loss for each of the head ouputs based on what was
        selected at initialization.

        Returns:
            dict: A dictionary with keys loss_cls, loss_bbox, and loss_energy.
        """
        assert len(label_preds) == len(bbox_preds) == len(energy_preds)

        feat_dims = [level.shape[-2:] for level in label_preds]
        all_level_points = []
        for i in range(len(feat_dims)):
            all_level_points.append(
                self.get_points_single(feat_dims[i], self.strides[i],
                                       bbox_preds[0].dtype,
                                       bbox_preds[0].device)
            )

        # First create targets
        bbox_targets, label_targets, energy_targets, mask = self.get_targets(
            gt_bboxes, gt_labels, feat_dims, img_metas)

        # Now reorder the targets so that they're usable and in the same
        # shape as the network outputs.
        bbox_targets, label_targets, energy_targets, mask = \
            self.reorder_targets(
                bbox_targets, label_targets, energy_targets, mask
            )

        # Then calculate energy losses.
        loss_energy = self.loss_energy(
            energy_preds, energy_targets.to(dtype=torch.long)
        )

        # Only consider loss for bboxes and labels_list at positions where the
        # energy is non-zero.
        pos_points = all_level_points[mask]
        pos_bbox_preds = bbox_preds[mask]
        pos_bbox_targets = bbox_targets[mask]
        pos_label_preds = label_preds[mask]
        pos_label_targets = label_targets[mask]

        loss_bbox = self.loss_bbox(
            distance2bbox(pos_points, pos_bbox_preds),
            pos_bbox_targets
        )
        loss_cls = self.loss_cls(pos_label_preds, pos_label_targets)

        if pos_bbox_preds.nelement() > 0:
            pass
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_cls = pos_label_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_energy=loss_energy,
        )

    @staticmethod
    def get_points_single(featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, gt_bboxes_list, gt_labels_list,
                    feat_dims, img_metas):
        """Gets targets for each output type.

        This method is also responsible for splitting the targets up into the
        the separate feature levels, i.e. figures out in which feature level to
        detect each object.

        The output returns labels_list, bboxes, and energy_preds. The bboxes will
        first be split based on max edge size. Then the energy_preds for each
        feature level will be calculated. Finally, the labels_list assigned to
        non-zero energy areas within each bounding box. All other areas will
        contain no label.

        Args:
            gt_bboxes_list (list): A list of tensors containing the ground
                truth bounding boxes.
            gt_labels_list (list): A list of tensors containing the ground
                truth labels_list of each bounding box.
            feat_dims (list): A list of 2-tuples where each element is the
                (h, w)
            img_metas (list): The img_metas as returned from the data loader.

        Returns:
            tuple: A tuple of bboxes, labels_list, energy_preds, and masks.
                This will be split first by image then by feature level,
                i.e. labels_list will be a list n lists, where n is the number
                of images, and each of those lists will have s elements, where s
                is the number of feature levels/heads. Each of those lists will
                then hold n tensors, n being the number of feature levels, and
                each tensor being the labels_list that are assigned to that
                feature level.

                bboxes will have shape (h, w, 4)
                labels will have shape (h, w)
                energy will have shape (h, w)
                masks will have shape (h, w) and is a boolean tensor

        """
        assert len(gt_bboxes_list) == len(gt_labels_list)

        # Sort gt_bboxes_list by object max edge
        split_bboxes = self.split_bboxes(gt_bboxes_list, gt_labels_list)

        # Calculate energy_preds for image for each feature level
        gt_bboxes = []
        gt_energy = []
        gt_labels = []
        gt_masks = []
        for i, bboxes in enumerate(split_bboxes):
            image_energy = []
            image_classes = []
            image_bboxes = []
            image_masks = []
            for j, feat_level_bboxes in enumerate(bboxes):
                img_size = img_metas[i]['pad_shape']

                feature_energy = self.get_energy_single(feat_dims[j],
                                                        img_size,
                                                        feat_level_bboxes)
                image_energy.append(feature_energy.values)
                # Using the image_energy, create a mask of background areas
                feature_mask = torch.where(feature_energy.values > 0,
                                           torch.tensor(1),
                                           torch.tensor(0)) == 1

                image_masks.append(feature_mask)

                # Then, using feature_energy.indices, get the class for each
                # grid cell that isn't background Entire area of non-zero
                # energy within a single bounding box should have the same
                # label.
                feature_classes = torch.zeros_like(feature_mask,
                                                   dtype=torch.float)
                feature_classes[feature_mask] = (
                    feat_level_bboxes[feature_energy.indices[feature_mask]]
                    [:, -1]
                )
                image_classes.append(feature_classes)

                # Finally, also assign bounding box values
                feature_bboxes = torch.zeros([feat_dims[j][0],
                                              feat_dims[j][1],
                                              4],
                                             dtype=torch.float,
                                             device=feat_level_bboxes.device)
                feature_bboxes[feature_mask] = (
                    feat_level_bboxes[feature_energy.indices[feature_mask]]
                    [:, 0:4]
                )
                image_bboxes.append(feature_bboxes)

            gt_energy.append(image_energy)
            gt_labels.append(image_classes)
            gt_bboxes.append(image_bboxes)
            gt_masks.append(image_masks)

        return gt_bboxes, gt_labels, gt_energy, gt_masks

    def split_bboxes(self, bbox_list, labels_list):
        """Splits bboxes based on max edge length.

        Args:
            bbox_list (list): The list of bounding boxes to be sorted. The
                list contains b tensors, where b is the batch size. Each
                tensor must be in the shape (n, 4) where n is the number of
                bounding boxes.
            labels_list (list): The list of ground truth labels associated with
                the bounding boxes. The list contains b tensors, where b is
                the batch size. Each tensor must be in the shape (n) where n
                is the equivalent to the number of bounding boxes.

        Returns:
            list: A list of length b, each element being a list of length s,
                where s is the number of heads used in the network. Each of
                these lists contains an (n, 5) tensor, which represents the
                bounding boxes with dim 5 being the class.
        """
        # max_indices is a 2 dim tensor, where dim 0 is the sorted indices
        # and dim 1 is the max_edge value
        max_indices = self.sort_bboxes(bbox_list)

        # Future TODO: Try sorting by area and see if that works better
        # TODO: Figure out what to do with background class.
        # Then move them to the appropriate level based on strides. The edge
        # size for each level should be [prev_regress_range, regress_range).
        # e.g. If we have ranges((-1, 4), (4, 8), (8, INF)), then we have edge
        # sizes [-1, 4), [4, 8), [8, INF)
        #
        # First split the max_indices tensor based on the values
        level_max_edge_lengths = [regress_range[0] for regress_range in
                                  self.regress_ranges]

        split_inds = []
        for max_index in max_indices:
            indices = []
            for length in level_max_edge_lengths:
                val = (max_index[1] > length).nonzero()
                val = val[0].item() if val is not None else indices[-1]
                indices.append(val)

            indices.append(len(max_index[1]))

            # indices is now the indices of the elements as split by max_edge,
            # split properly into each feature level.
            #
            # We now split the actual bboxes into the values
            split_inds.append([max_index[0][indices[i] : indices[i + 1]]
                               for i in range(len(indices) - 1)])

            # split_bbox_ind is appended an s length list, where each element
            # contains all the indices that belong to that feature level.

        out_list = []
        for i in range(len(bbox_list)):             # Iterate through each image
            temp_list = []
            for inds in split_inds[i]:                 # Iterate through head
                # Grab bboxes with the given indices, then the labels
                bbox = bbox_list[i][inds.to(dtype=torch.long)]
                labels = labels_list[i][inds.to(dtype=torch.long)].to(
                    dtype=torch.float)
                # Labels must be unsqueezed to allow concatenation
                temp_list.append(
                    torch.cat((bbox, labels.unsqueeze(1)), dim=1)
                )
            out_list.append(temp_list)

        return out_list

    @staticmethod
    def sort_bboxes(bbox_list):
        """Sorts bboxes based on max_edge length.

        Args:
            bbox_list (list): The list of bounding boxes to be sorted. The
                bounding boxes must be tensors in the shape (n, 4).

        Returns:
            list: A list of (2, n) tensors, where n is the number of bboxes
                and 2 being the indice and max edge length of the
                corresponding tensor.
        """
        out_list = []
        for bboxes in bbox_list:
            edges = torch.cat((bboxes[:, 2] - bboxes[:, 0],
                               bboxes[:, 3] - bboxes[:, 1]))

            # Split to a 2-dim array, dim 0 being the x length and dim 1
            # being the y length
            edges = edges.reshape(2, bboxes.shape[0])

            # Then transpose it to associate both x and y with the same
            # value. This is done simply because it is conceptually easier to
            # understand.
            edges = edges.transpose(0, 1)

            # Get the max, then get the sorted indices.
            max_edges = edges.max(1).values
            sorted_inds = max_edges.argsort()

            # Concatenate them and add them to the out_list
            out_list.append(torch.cat((sorted_inds.to(dtype=torch.float),
                                       max_edges[sorted_inds]))
                            .reshape(2, bboxes.shape[0]))

        return out_list

    def get_energy_single(self, feat_dim, img_size, bboxes):
        """Gets energy for a single feature level based on deep watershed.

        Args:
            feat_dim (tuple): A 2-tuple containing the height and width of the
                current feature level. (h, w)
            img_size (tuple): A 2-tuple containg the size of the image. Used
                for scaling the bboxes to the feature level dimensions. (h, w)
            bboxes (torch.Tensor): A tensor of the bboxes that belong the this
                feature level with shape (n, 4).

        Notes:
            The energy targets are calculated as:
            E_max \cdot argmax_{c \in C}[1 - \sqrt{((l-r) / 2)^2
                                                   + ((t-b) / 2)^2}
                                         / r]

            - r is a hyperparameter we would like to minimize.
            - (l-r)/2 is the horizontal distance to the center and will be
                assigned the variable name "horizontal"
            - (t-b)/2 is the vertical distance to the center and will be
                assigned the variable name "vertical"
            - E_max is self.max_energy

        Returns:
            torch.return_types.max: The max energy values and the bounding
                box they belong to.
        """
        # TODO UPDATE DOCUMENTATION
        type_dict = {'dtype': bboxes.dtype, 'device': bboxes.device}

        # First create an n dimensional tensor, where n is the number of bboxes
        energy_layers = torch.zeros([bboxes.shape[0], feat_dim[0], feat_dim[1]],
                                    **type_dict)
        zero_tensor = torch.tensor(0., **type_dict)

        # Now cast each bbox to each cell in the energy layer that it covers
        # First bounds of grid squares that have a bbox in them
        x_scale_factor = feat_dim[1] / img_size[1]
        y_scale_factor = feat_dim[0] / img_size[0]
        scale_factor = torch.tensor((x_scale_factor, y_scale_factor,
                                     x_scale_factor, y_scale_factor,
                                     1.), **type_dict)
        scale_factor = scale_factor.repeat(bboxes.shape[0], 1)
        adder = torch.tensor((0, 0, 1, 1, 0), device=type_dict['device'])
        adder = adder.repeat(bboxes.shape[0], 1)

        grid_bounds = torch.floor(bboxes * scale_factor).long() + adder

        x_index = torch.arange(0, feat_dim[1], **type_dict).repeat(
            feat_dim[0], 1)
        y_index = torch.arange(0, feat_dim[0], **type_dict).repeat(
            feat_dim[1], 1).transpose(0, 1)


        # Fill each energy layer
        for energy_layer, grid_bound, bbox in zip(energy_layers, grid_bounds,
                                                  bboxes):
            # Go through each bbox. First create the mask of grid areas where
            # the bounding box exists.
            mask = torch.zeros_like(energy_layer).to(dtype=torch.long)
            mask[grid_bound[1]:grid_bound[3], grid_bound[0]:grid_bound[2]] = 1
            mask.bool()

            bbox_dist = torch.tensor((bbox[0] + bbox[2],
                                      bbox[1] + bbox[3]),
                                     **type_dict)

            # This is basically direct from the math formulation designed to run
            # in a vectorized manner.
            horizontal = (bbox_dist[0] - (2. * x_index[mask == 1]
                                          / x_scale_factor)) / 2
            vertical = (bbox_dist[1] - (2. * y_index[mask == 1]
                                        / y_scale_factor)) / 2

            # Multiplied by self is faster than tensor.pow(2) by about 30%
            val = (horizontal * horizontal) + (vertical * vertical)
            val = 1 - (torch.sqrt(val) / self.r)
            val = torch.floor(temp * self.max_energy)

            # torch.max to eliminate negative numbers. torch.max is
            # approximately 20 times faster than using indexing
            temp = torch.max(temp, zero_tensor)

            energy_layer[mask == 1] = torch.floor(temp * self.max_energy)

        return energy_layers.max(dim=0)

    @staticmethod
    def reorder_targets(bbox_targets, label_targets, energy_targets,
                        masks_targets):
        """Reorders targets such that they are the same shape as predictions.

        Notes:
            b stands for batch size and s stands for number of feature
            levels/heads.

        Shapes:
            bbox_targets: b-list of s-lists of (h, w, 4) tensors.
            label_targets: b-list of s-lists of (h, w) tensors.
            energy_target: b-list of s-lists of (h, w) tensors.
            masks: b-list of s-lists of (h, w) tensors.
        Args:
             bbox_targets (list): List of bbox targets from get_targets()
             label_targets (list): List of label targets from get_targets()
             energy_targets (list): List of energy targets from get_targets()
             masks (list): List of masks of non-zero energy from get_targets().
                This is a boolean tensor.

         Returns:
             list: s-list of (b, h, w, 4) tensors representing bboxes.
             list: s-list of (b, h, w) tensors representing labels.
             list: s-list of (b, h, w) tensors representing energy_preds.
             list: s-list of (b, h, w) boolean tensors representing the non-zero
                masks.
        """
        bboxes = [[] for _ in range(len(bbox_targets[0]))]
        labels = [[] for _ in range(len(bbox_targets[0]))]
        energy = [[] for _ in range(len(bbox_targets[0]))]
        masks = [[] for _ in range(len(bbox_targets[0]))]

        for image_num in range(len(bbox_targets)):
            for i, (b_target, l_target, e_target, m_target) in enumerate(
                zip(bbox_targets[image_num],
                    label_targets[image_num],
                    energy_targets[image_num],
                    masks_targets[image_num])):
                bboxes[i].append(torch.unsqueeze(b_target, 0))
                labels[i].append(torch.unsqueeze(l_target, 0))
                energy[i].append(torch.unsqueeze(e_target, 0))
                masks[i].append(torch.unsqueeze(m_target, 0))

        for i in range(len(bbox_targets[0])):
            bboxes[i] = torch.cat(bboxes[i])
            labels[i] = torch.cat(labels[i])
            energy[i] = torch.cat(energy[i])
            masks[i] = torch.cat(masks[i])

        return bboxes, labels, energy, masks

    def get_pos_only(self, bbox_preds, bbox_targets,
                     label_preds, label_targets, mask):
        """Gets only the bboxes and labels where the mask is true.

        Returns:
            torch.Tensor: (d, 4) positive bbox predictions.
            torch.Tensor: (d, 4) positive bbox targets.
            torch.Tensor: (d, num_classes) positive label predictions.
            torch.Tensor: (d) positive label targets.
        """
        return (bbox_preds[mask])

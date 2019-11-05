import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob

INF = 1e8


@HEADS.register_module
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
                 r=5.):
        """
        Creates a head based on FCOS that uses an energies map, not centerness
        Args:
            num_classes (int): Number of classes to output.
            in_channels (int): Number of innput channels.
            max_energy (int): Quantization of energies. How much to split the
                energies values by.
            feat_channels (int): Number of feature channels in each of the
                stacked convolutions.
            stacked_convs (int): Number of stacked convolutions to have.
            strides (tuple): Stride value for each of the heads.
            regress_ranges (tuple): The regression range for each of the heads.
            loss_cls (dict): A description of the loss to use for the
                classfication output.
            loss_bbox (dict): A description of the loss to use for the bbox
                output.
            loss_energy (dict): A description of the loss to use for the energies
                map output.
            conv_cfg (dict): A description of the configuration of the
                convolutions in the stacked convolution.
            norm_cfg (dict): A description of the normalization configuration of
                the layers of the stacked convolution.
            split_convs (bool): Whether or not to split the classification and
                energies map convolution stacks. False means that the
                classification energies map shares the same convolution stack.
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
        self.num_classes = num_classes
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
            self.feat_channels, self.cls_out_channels, 3, padding=1)
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
                cls_score, bbox_pred, energies of the different feature layers.
        """
        # Use a multi_apply function to run forwards on each feats tensor
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
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

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'energies'))
    def loss(self,
             cls_scores,
             bbox_preds,
             energies,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(energies)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.wfcos_target(all_level_points, gt_bboxes,
                                                 gt_labels)



        # Labels are a list of per level labels, each level is a tensor of all
        # targets at that level

        # print("bbox_targets[0].shape: {}".format(bbox_targets[0].shape))
        # print("bbox_target length: {}".format(len(bbox_targets)))

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and energies
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]

        # Calculate flattened energies
        flatten_energies = []
        for energy in energies:
            energy = energy.permute(0, 2, 3, 1)
            es = energy.shape  # Easier access
            flatten_energies.append(energy.reshape([es[0] * es[1] * es[2],
                                                    self.max_energy]))
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_energies = torch.cat(flatten_energies, dim=0)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        # print("pos_inds.shape: {}".format(pos_inds.shape))
        # print("pos_inds: {}".format(pos_inds))
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_energies = flatten_energies[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_energies_targets, energies_targets = self.energy_target(
                flatten_bbox_targets, pos_bbox_targets, pos_inds)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            if pos_energies_targets is not None \
                and not torch.any(pos_energies_targets > 0):
                pos_energies_targets = pos_energies_targets.reshape(
                    [pos_energies_targets.shape[0], 1]).repeat(1, 4)

            # print("energies_targets \n{}\n".format(pos_energies_targets))
            # print("energies: \n{}\n".format(pos_energies))
            # energy weighted iou loss

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_energies_targets,
                avg_factor=pos_energies_targets.sum())
            loss_energy = self.loss_energy(flatten_energies,
                                           energies_targets
                                           .to(dtype=torch.long))
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_energy = pos_energies.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_energy=loss_energy)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'energies'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   energies,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)


        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            energy_pred_list = [
                energies[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                energy_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)

        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          energies,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_energy = []
        for cls_score, bbox_pred, energy, points in zip(
                cls_scores, bbox_preds, energies, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # Flatten to argmax values
            energy = energy.permute(1, 2, 0).argmax(2).reshape(-1)

            # Turn it into a float
            energy = energy.to(dtype=torch.float32)

            # Then apply a sigmoid function to it before increasing back to the
            # max energy level
            energy = energy.div(self.max_energy).sigmoid()

            # Finally floor it
            energy = energy.floor()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * energy[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                energy = energy[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_energy.append(energy)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_energy = torch.cat(mlvl_energy)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_energy)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def wfcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.wfcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def wfcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def energy_target(self, flattened_bbox_targets, pos_bbox_targets,
                      pos_indices):
        """Calculate energy targets based on deep watershed paper.

        Args:
            flattened_bbox_targets (torch.Tensor): The flattened bbox targets.
            pos_bbox_targets (torch.Tensor): Bounding box lrtb values only for
                positions within the bounding box. We use this as an argument
                to prevent recalculating it since it is used for other things as
                well.
            pos_indices (torch.Tensor): The indices of values in
                flattened_bbox_targets which are within a bounding box

        Notes:
            The energy targets are calculated as:
            E_max \cdot argmax_{c \in C}[1 - \sqrt{((l-r)/2)^2 + ((t-b) / 2)^2}
                                         / r]

            - r is a hyperparameter we would like to minimize.
            - (l-r)/2 is the horizontal distance to the center and will be
                assigned the variable name "horizontal"
            - (t-b)/2 is the vertical distance to the center and will be
                assigned the variable name "vertical"
            - E_max is self.max_energy
            - We don't need the argmax in this code implementation since we
                already select the bounding boxes and their respective pixels in
                a previous step.

        Returns:
            tuple: A 2 tuple with values ("pos_energies_targets",
                "energies_targets"). Both are flattened but pos_energies_targets
                only contains values within bounding boxes.
        """

        horizontal = pos_bbox_targets[:, 0] - pos_bbox_targets[:, 2]
        vertical = pos_bbox_targets[:, 1] - pos_bbox_targets[:, 3]

        # print("Horizontals: {}".format(horizontal))
        # print("Verticals: {}".format(vertical))

        horizontal = torch.div(horizontal, 2)
        vertical = torch.div(vertical, 2)

        c2 = (horizontal * horizontal) + (vertical * vertical)

        # print("c2: \n{}".format(c2))

        # We use x * x instead of x.pow(2) since it's faster by about 30%
        square_root = torch.sqrt(c2)

        # print("Sqrt: \n{}".format(square_root))

        type_dict = {'dtype': square_root.dtype,
                     'device': square_root.device}

        pos_energies = (torch.tensor([1], **type_dict)
                        - torch.div(square_root, self.r))
        pos_energies *= self.max_energy
        pos_energies = torch.max(pos_energies,
                                 torch.tensor([0], **type_dict))
        pos_energies = pos_energies.floor()

        energies_targets = torch.zeros(flattened_bbox_targets.shape[0],
                                       **type_dict)
        energies_targets[pos_indices] = pos_energies

        # torch.set_printoptions(profile='full')
        # print("Energy targets: \n {}".format(pos_energies))
        # torch.set_printoptions(profile='default')
        # input()

        return pos_energies, energies_targets

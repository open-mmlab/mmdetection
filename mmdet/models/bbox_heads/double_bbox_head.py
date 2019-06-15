import torch.nn as nn

from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
from ..plugins import FPNUpChannels, FPNFFConv 

@HEADS.register_module
class DoubleConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    ## Old
    #                             /-> cls convs -> cls fcs -> cls
    # shared convs -> shared fcs
    #                             \-> reg convs -> reg fcs -> reg

    ## New
                                      /-> cls
                  /-> shared convs -> 
                                      \-> reg
    roi features     
                                      /-> cls
                  \-> shared fc     ->
                                      \-> reg

    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(DoubleConvFCBBoxHead, self).__init__(*args, **kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        ## input 256
        ## output 1024
        self.nonlocal_conv = FPNUpChannels(conv_out_channels, 1024)
        self.conv_inter_channels = 1024

        # add conv heads
        self.shared_convs, last_layer_dim = \
            self._add_conv_nonlocal_branch(
                self.num_shared_convs, self.conv_inter_channels
                )
        self.shared_conv_out_channels = last_layer_dim


        # add fc heads
        self.shared_fcs, last_layer_dim = \
            self._add_fc_branch(
                self.num_shared_fcs, self.in_channels
                )
        self.shared_fc_out_channels = last_layer_dim

        ## self.in_channels = 256 defined in bbox_head
        self.with_avg_pool = True
        self.avg_pool = nn.AvgPool2d(self.roi_feat_size)

        self.conv_last_dim = self.shared_conv_out_channels
        self.fc_last_dim = self.shared_fc_out_channels

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed

        ## conv cls & reg
        out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                       self.num_classes)
        # self.conv_cls = nn.Linear(self.conv_last_dim, self.num_classes)
        # self.conv_reg = nn.Linear(self.conv_last_dim, out_dim_reg)
        ## roll back to fc_reg 
        self.fc_reg = nn.Linear(self.conv_last_dim, out_dim_reg)

        self.fc_cls = nn.Linear(self.fc_last_dim, self.num_classes)
        # self.fc_reg = nn.Linear(self.fc_last_dim, out_dim_reg)

    def _add_conv_nonlocal_branch(self,
                            num_branch_convs,
                            in_channels):
        """Add shared or separable branch

        convs -> avg pool 
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_inter_channels)

                branch_convs.append(
                    FPNFFConv(in_channels=last_layer_dim) 
                    # NONLocalBlock2D_Group(in_channels=last_layer_dim, num_group=4) 
                    )
            last_layer_dim = self.conv_inter_channels 

        return branch_convs, last_layer_dim

    def _add_fc_branch(self,
                            num_branch_fcs,
                            in_channels ):
        """Add shared or separable branch
        fcs
        """
        last_layer_dim = in_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for separated branches, also consider self.num_shared_fcs
            ## flatten 
            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim

    def init_weights(self):
        # super(ConvFCBBoxHead, self).init_weights()
        ## conv cls
        # nn.init.normal_(self.conv_cls.weight, 0, 0.01)
        # nn.init.constant_(self.conv_cls.bias, 0)
        ## conv reg
        # nn.init.normal_(self.conv_reg.weight, 0, 0.001)
        # nn.init.constant_(self.conv_reg.bias, 0)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)
        ## fc cls
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)
        ## fc reg
        # nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        # nn.init.constant_(self.fc_reg.bias, 0)

        # for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
        for module_list in [self.shared_convs, self.shared_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    ## reconstruct losses
    # def loss(self, 
    #          conv_cls_score,
    #          conv_bbox_pred,
    #          fc_cls_score,
    #          fc_bbox_pred,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduce=True):
    #     losses = dict()
    #     if conv_cls_score is not None:
    #         losses['loss_conv_cls'] = self.loss_cls(
    #             conv_cls_score, labels, label_weights, reduce=reduce)
    #         losses['conv_acc'] = accuracy(conv_cls_score, labels)
    #     if fc_cls_score is not None:
    #         losses['loss_fc_cls'] = self.loss_cls(
    #             fc_cls_score, labels, label_weights, reduce=reduce)
    #         losses['fc_acc'] = accuracy(fc_cls_score, labels)

    #     if conv_bbox_pred is not None:
    #         pos_inds = labels > 0
    #         if self.reg_class_agnostic:
    #             conv_pos_bbox_pred = conv_bbox_pred.view(conv_bbox_pred.size(0), 4)[pos_inds]
    #         else:
    #             conv_pos_bbox_pred = conv_bbox_pred.view(conv_bbox_pred.size(0), -1,
    #                                            4)[pos_inds, labels[pos_inds]]

    #         losses['loss_conv_bbox'] = self.loss_bbox(
    #             conv_pos_bbox_pred,
    #             bbox_targets[pos_inds],
    #             bbox_weights[pos_inds],
    #             avg_factor=bbox_targets.size(0))

    #     if fc_bbox_pred is not None:
    #         pos_inds = labels > 0
    #         if self.reg_class_agnostic:
    #             fc_pos_bbox_pred = fc_bbox_pred.view(fc_bbox_pred.size(0), 4)[pos_inds]
    #         else:
    #             fc_pos_bbox_pred = fc_bbox_pred.view(fc_bbox_pred.size(0), -1,
    #                                            4)[pos_inds, labels[pos_inds]]
    #         losses['loss_fc_bbox'] = self.loss_bbox(
    #             fc_pos_bbox_pred,
    #             bbox_targets[pos_inds],
    #             bbox_weights[pos_inds],
    #             avg_factor=bbox_targets.size(0))

    #     return losses

    def forward(self, x):
        # shared part
        x_conv = x
        x_fc = x

        ## conv head
        x_conv = self.nonlocal_conv(x_conv)

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        # conv_cls_score = self.conv_cls(x_conv) 
        # conv_bbox_pred = self.conv_reg(x_conv) 
        conv_bbox_pred = self.fc_reg(x_conv) 

        ## fc head
        if self.num_shared_fcs > 0:
            x_fc = x_fc.view(x_fc.size(0), -1)
            for fc in self.shared_fcs:
                x_fc = self.relu(fc(x_fc))

        fc_cls_score = self.fc_cls(x_fc) 
        # fc_bbox_pred = self.fc_reg(x_fc) 

        # return conv_cls_score, conv_bbox_pred, fc_cls_score, fc_bbox_pred
        return fc_cls_score, conv_bbox_pred


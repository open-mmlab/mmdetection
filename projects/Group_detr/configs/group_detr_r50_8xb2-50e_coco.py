_base_ = ['mmdet::conditional_detr/conditional-detr_r50_8xb2-50e_coco.py']

custom_imports = dict(
    imports=['projects.Group_detr.group_detr'], allow_failed_imports=False)

# The number of decoder query groups.
num_query_groups = 11

model = dict(
    type='GroupDETR',
    num_query_groups=num_query_groups,
    decoder=dict(layer_cfg=dict(self_attn_cfg=dict(group_detr=num_query_groups))),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GHungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ],
            group_detr=num_query_groups)))

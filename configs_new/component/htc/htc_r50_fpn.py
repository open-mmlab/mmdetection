_base_ = './htc_without_semantic_r50_fpn.py'
# model settings
model = dict(
    semantic_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[8]),
    semantic_head=dict(
        type='FusedSemanticHead',
        num_ins=5,
        fusion_level=1,
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=183,
        ignore_label=255,
        loss_weight=0.2))

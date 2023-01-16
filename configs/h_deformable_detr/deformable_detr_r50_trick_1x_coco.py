_base_ = '../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py'  # noqa
model = dict(
    bbox_head=dict(
        mixed_selection=True,
        transformer=dict(
            encoder=dict(
                transformerlayers=dict(
                    attn_cfgs=dict(dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0)),
            decoder=dict(
                look_forward_twice=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0)))))
data = dict(samples_per_gpu=4)
# learning policy
lr_config = dict(policy='step', step=[11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

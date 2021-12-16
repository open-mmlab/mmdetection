"""This version was mentioned in Table XI, in paper `Masked-attention Mask
Transformer for Universal Image
Segmentation<https://arxiv.org/abs/2112.01527>`_"""

_base_ = './maskformer_r50_mstrain_64x1_300e_coco.py'

# optimizer
optimizer = dict(weight_decay=0.0005)

# learning policy
lr_config = dict(step=[50])
runner = dict(type='EpochBasedRunner', max_epochs=75)

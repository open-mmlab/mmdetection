_base_ = './yolof_r50_c5_8x8_1x_coco.py'

# We implemented the iter-based config according to the source code.
# COCO dataset has 117266 images after filtering. We use 8 gpu and
# 8 batch size training, so 22500 is equivalent to
# 22500/(117266/(8x8))=12.3 epoch, 15000 is equivalent to 8.2 epoch,
# 20000 is equivalent to 10.9 epoch. Due to lr(0.12) is large,
# the iter-based and epoch-based setting have about 0.2 difference on
# the mAP evaluation value.
lr_config = dict(step=[15000, 20000])
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=22500)
checkpoint_config = dict(interval=2500)
evaluation = dict(interval=4500)
log_config = dict(interval=20)

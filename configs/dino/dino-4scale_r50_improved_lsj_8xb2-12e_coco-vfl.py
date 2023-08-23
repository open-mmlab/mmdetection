_base_ = ['dino-4scale_r50_improved_lsj_8xb2-12e_coco.py']

model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='VarifocalLoss',
            use_sigmoid=True,
            loss_weight=2.0)))

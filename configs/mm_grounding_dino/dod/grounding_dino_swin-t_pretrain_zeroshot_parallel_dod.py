_base_ = 'grounding_dino_swin-t_pretrain_zeroshot_concat_dod.py'

model = dict(test_cfg=dict(chunked_size=1))

_base_ = 'grounding_dino_swin-l_pretrain_zeroshot_concat_dod.py'

model = dict(test_cfg=dict(chunked_size=1))

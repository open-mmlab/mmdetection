_base_ = '../../configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'

runner_type = 'FlexibleRunner'

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.size_based_auto_wrap_policy',
            min_num_params=1e7)))


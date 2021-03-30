import torch
from torch.onnx import is_in_onnx_export


def batched_nms_with_extra_nms_args(bboxes, scores, labels, nms_cfg):
    """
    This function updates 'score_threshold' and 'max_num' in the next 'NMSop' call. 
    It should be replaces with 'mmcv::batched_nms' after adding support for 'score_threshold' and 'max_num' in MMCV.
    """
    if torch.onnx.is_in_onnx_export():
        score_thr = nms_cfg.pop('score_thr') if 'score_thr' in nms_cfg.keys() else 0.0
        max_num = nms_cfg.pop('max_num') if 'max_num' in nms_cfg.keys() else int(bboxes.size(0))
    
        from ...utils.deployment.symbolic import set_extra_args_for_NMSop
        set_extra_args_for_NMSop(score_thr, max_num)
    else:
        for key in ['score_thr', 'max_num', 'type']:  # Remain only 'iou_threshold'
            if key in nms_cfg.keys():
                nms_cfg.pop(key)

    from mmcv.ops import batched_nms
    return batched_nms(bboxes, scores, labels, nms_cfg)

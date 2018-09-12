from mmdet.datasets import collate
from mmdet.nn.parallel import scatter

__all__ = ['_data_func']

def _data_func(data, gpu_id):
    imgs, img_metas = tuple(
        scatter(collate([data], samples_per_gpu=1), [gpu_id])[0])
    return dict(
        img=imgs,
        img_meta=img_metas,
        return_loss=False,
        return_bboxes=True,
        rescale=True)

from mmengine.config import Config

from mmdet.registry import MODELS


def print_model():
    cfg_path = './projects/OneFormer/configs/oneformer_r50_lsj_8x2_50e_coco-panoptic.py'
    # cfg_path = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
    model_cfg = Config.fromfile(cfg_path)
    detector = MODELS.build(model_cfg['model'])
    for ind, i in detector.state_dict().items():
        print(ind, i.shape)


if __name__ == '__main__':
    print_model()

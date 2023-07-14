from mmengine.config import Config

from mmdet.registry import MODELS
import torch


def print_model():
    cfg_path = './projects/OneFormer/configs/oneformer_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py'
    # cfg_path = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
    model_cfg = Config.fromfile(cfg_path)
    detector = MODELS.build(model_cfg['model'])
    with open('one_former.txt', 'w') as f:
        for ind, i in detector.state_dict().items():
            print(ind, i.shape)
            f.write(ind + '------>')
            f.write(str(i.shape) + '\n')

def print_checkpoint():
    pth_path = '150_16_swin_l_oneformer_coco_100ep.pth'
    # cfg_path = 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
    oneformer_model = torch.load(pth_path)
    detector = oneformer_model
    with open('converted_one_former.txt', 'w') as f:
        for ind, i in detector['state_dict'].items():
            print(ind, i.shape)
            f.write(ind + '------>')
            f.write(str(i.shape) + '\n')


if __name__ == '__main__':
    print_model()

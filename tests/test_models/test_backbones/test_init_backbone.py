from mmcv import Config
from mmcv.runner import _load_checkpoint

from mmdet.models import build_detector

configs = [
    'configs/atss/atss_r50_fpn_1x_coco.py',
    'configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py',
    'configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py',
    'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py',
    'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
    'configs/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py',
    # 'configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py',
    # do not use pretrained
    # 'configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py',
    # do not use pretrained
    'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py',
    'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
    'configs/detectors/detectors_htc_r50_1x_coco.py',
    'configs/detr/detr_r50_8x2_150e_coco.py',
    'configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py',
    'configs/empirical_attention/'
    'faster_rcnn_r50_fpn_attention_1111_1x_coco.py',
    'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    'configs/fcos/fcos_center-normbbox-centeronreg-giou'
    '_r50_caffe_fpn_gn-head_1x_coco.py',
    'configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py',
    'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
    'configs/fsaf/fsaf_r50_fpn_1x_coco.py',
    'configs/gcnet/mask_rcnn_r50_fpn_'
    'syncbn-backbone_r16_gcb_c3-c5_1x_coco.py',
    'configs/gfl/gfl_r50_fpn_1x_coco.py',
    'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py',
    'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py',
    'configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py',
    'configs/groie/faster_rcnn_r50_fpn_groie_1x_coco.py',
    'configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py',
    'configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py',
    'configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py',
    'configs/htc/htc_r50_fpn_1x_coco.py',
    'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
    'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py',
    'configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py',
    'configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py',
    'configs/paa/paa_r50_fpn_1x_coco.py',
    'configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py',
    'configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py',
    'configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py',
    'configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py',
    'configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py',
    'configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+'
    'head_mstrain-range_1x_coco.py',
    'configs/retinanet/retinanet_r50_fpn_1x_coco.py',
    'configs/rpn/rpn_r50_fpn_1x_coco.py',
    'configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py',
    'configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py',
    'configs/scnet/scnet_r50_fpn_1x_coco.py',
    'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py',
    'configs/ssd/ssd300_coco.py',
    'configs/tridentnet/tridentnet_r50_caffe_1x_coco.py',
    'configs/vfnet/vfnet_r50_fpn_1x_coco.py',
    'configs/yolact/yolact_r50_1x8_coco.py',
    'configs/yolo/yolov3_d53_320_273e_coco.py',
    'configs/yolof/yolof_r50_c5_8x8_1x_coco.py',
    'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py'
]
for config in configs:
    # config = 'configs/yolo/yolov3_d53_320_273e_coco.py'
    print('----------------------------loading ', config)
    cfg = Config.fromfile(config)

    checkpoint = _load_checkpoint(cfg.model.backbone.init_cfg.checkpoint, )
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    layers = []
    names = []
    checkpoint_names = []
    layer_indexes = []
    darknet_indexs = []
    for name in model.backbone.state_dict():
        split_name = name.split('.')
        if split_name[-1] == 'weight':
            if 'stem' in split_name[0]:
                layers.append(split_name[0])
                layer_indexes.append(split_name[1])
                checkpoint_names.append(name)
                names.append(None)
                darknet_indexs.append(None)
            if 'features' in split_name[0]:
                names.append(None)
                checkpoint_names.append(name)
                layer_indexes.append(split_name[1])
                layers.append(split_name[0])
                darknet_indexs.append(None)
            if 'conv' in split_name[-2]:
                # conv
                if len(split_name) == 2:
                    # conv in backbone
                    names.append(split_name[-2])
                    checkpoint_names.append(name)
                    layer_indexes.append(None)
                    layers.append(None)
                    darknet_indexs.append(None)
                elif len(split_name) == 3:
                    # darknet
                    names.append(split_name[-2])
                    checkpoint_names.append(name)
                    layer_indexes.append(None)
                    layers.append(split_name[0])
                    darknet_indexs.append(None)
                    # vgg
                elif len(split_name) == 4:
                    if 'layer' in split_name[0]:
                        # resnet
                        names.append(split_name[-2])
                        checkpoint_names.append(name)
                        layer_indexes.append(split_name[1])
                        layers.append(split_name[0])
                        darknet_indexs.append(None)
                elif len(split_name) == 5:
                    if 'conv_res' in split_name[0]:
                        names.append(split_name[-2])
                        checkpoint_names.append(name)
                        layer_indexes.append(split_name[2])
                        layers.append(split_name[0])
                        darknet_indexs.append(split_name[1])

    assert len(names) > 0
    assert len(names) == len(checkpoint_names) == \
           len(layer_indexes) == len(layers) == len(darknet_indexs)
    for i in range(len(names)):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        weight_sum = state_dict[checkpoint_names[i]].sum()
        if layer_indexes[i] is None:
            if layers[i] is None:
                after_init_weight_sum = getattr(model.backbone,
                                                names[i]).weight.sum()
            else:
                layer = getattr(model.backbone, layers[i])
                after_init_weight_sum = getattr(layer, names[i]).weight.sum()
        else:
            if darknet_indexs[i] is not None:
                layer = getattr(model.backbone, layers[i])
                conv = getattr(layer, darknet_indexs[i])
                conv = getattr(conv, layer_indexes[i])
                after_init_weight_sum = getattr(conv, names[i]).weight.sum()
            else:
                if names[i] is None:
                    conv = getattr(model.backbone,
                                   layers[i])[int(layer_indexes[i])]
                    after_init_weight_sum = conv.weight.sum()
                else:
                    layer = getattr(model.backbone,
                                    layers[i])[int(layer_indexes[i])]
                    after_init_weight_sum = getattr(layer,
                                                    names[i]).weight.sum()
        assert weight_sum == after_init_weight_sum
    print('-----------------Successfully load checkpoint-----------------')
    # assert_convs = ['conv1']
    # for assert_conv in assert_convs:
    #     weight_sum = checkpoint['state_dict'][assert_conv+'.weight'].sum()
    #     after_init_weight_sum = getattr(model.backbone,
    #                                     assert_conv).weight.sum()
    #     assert weight_sum == after_init_weight_sum
    #     print()

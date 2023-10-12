# Changelog of v3.x

## v3.1.0 (12/10/2023)

### Highlights

**(1) Detection Transformer SOTA Model Collection**

- Supported four updated and stronger SOTA Transformer models: DDQ, CO-DETR, AlignDETR, and H-DINO.
- Based on CO-DETR, MMDet released a model with a COCO performance of 64.1 mAP.
- Algorithms such as DINO support AMP/Checkpoint/FrozenBN, which can effectively reduce memory usage.

**(2) Comprehensive Performance Comparison between CNN and Transformer**

RF100 consists of a dataset collection of 100 real-world datasets, including 7 domains. It can be used to assess the performance differences of Transformer models like DINO and CNN-based algorithms under different scenarios and data volumes. Users can utilize this benchmark to quickly evaluate the robustness of their algorithms in various scenarios.

**(3) Support for GLIP and Grounding DINO fine-tuning, the only algorithm library that supports Grounding DINO fine-tuning**

The Grounding DINO algorithm in MMDet is the only library that supports fine-tuning. Its performance is one point higher than the official version, and of course, GLIP also outperforms the official version.
We also provide a detailed process for training and evaluating Grounding DINO on custom datasets. Everyone is welcome to give it a try.

**(4) Support for the open-vocabulary detection algorithm Detic and multi-dataset joint training.**

**(5) Training detection models using FSDP and DeepSpeed.**

**(6) Support for the V3Det dataset, a large-scale detection dataset with over 13,000 categories.**

### New Features

- Support CO-DETR/DDQ/AlignDETR/H-DINO
- Support GLIP and Grounding DINO fine-tuning
- Support Detic and Multi-Datasets training (#10926)
- Support V3Det and benchmark (#10938)
- Support Roboflow 100 Benchmark (#10915)
- Add custom dataset of grounding dino (#11012)
- Release RTMDet-X p6 (#10993)
- Support AMP of DINO (#10827)
- Support FrozenBN (#10845)
- Add new configuration files for `QDTrack/DETR/RTMDet/MaskRCNN/DINO/DeformableDETR/MaskFormer` algorithm
- Add a new script to support the WBF (#10808)
- Add `large_image_demo` (#10719)
- Support download dataset from OpenXLab (#10799)
- Update to support torch2onnx for DETR series models (#10910)
- Translation into Chinese of an English document (#10744, #10756, #10805, #10848)

### Bug Fixes

- Fix name error in DETR metafile.yml (#10595)
- Fix device of the tensors in `set_nms` (#10574)
- Remove some unicode chars from `en/` docs (#10648)
- Fix download dataset with mim script. (#10727)
- Fix export to torchserve (#10694)
- Fix typo in `mask-rcnn_r50_fpn_1x-wandb_coco` (#10757)
- Fix `eval_recalls` error in `voc_metric` (#10770)
- Fix torch version comparison (#10934)
- Fix incorrect behavior to access train pipeline from ConcatDataset in `analyze_results.py` (#11004)

### Improvements

- Update `useful_tools.md` (#10587)
- Update Instance segmentation Tutorial (#10711)
- Update `train.py` to compat with new config (#11025)
- Support `torch2onnx` for maskformer series (#10782)

### Contributors

A total of 36 developers contributed to this release.

Thank @YQisme, @nskostas, @max-unfinity, @evdcush, @Xiangxu-0103, @ZhaoCake, @RangeKing, @captainIT, @ODAncona, @aaronzs, @zeyuanyin, @gotjd709, @Musiyuan, @YanxingLiu, @RunningLeon, @ytzfhqs, @zhangzhidaSunny, @yeungkong, @crazysteeaam, @timerring, @okotaku, @apatsekin, @Morty-Xu, @Markson-Young, @ZhaoQiiii, @Kuro96, @PhoenixZ810, @yhcao6, @myownskyW7, @jiongjiongli, @Johnson-Wang, @ryylcc, @guyleaf, @agpeshal, @SimonGuoNjust, @hhaAndroid

## v3.1.0 (30/6/2023)

### Highlights

- Supports tracking algorithms including multi-object tracking (MOT) algorithms SORT, DeepSORT, StrongSORT, OCSORT, ByteTrack, QDTrack, and video instance segmentation (VIS) algorithm MaskTrackRCNN, Mask2Former-VIS.
- Support [ViTDet](../../../projects/ViTDet)
- Supports inference and evaluation of multimodal algorithms [GLIP](../../../configs/glip) and [XDecoder](../../../projects/XDecoder), and also supports datasets such as COCO semantic segmentation, COCO Caption, ADE20k general segmentation, and RefCOCO. GLIP fine-tuning will be supported in the future.
- Provides a [gradio demo](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/projects/gradio_demo/README.md) for image type tasks of MMDetection, making it easy for users to experience.

### New Features

- Support DSDL Dataset (#9801)
- Support iSAID dataset (#10028)
- Support VISION dataset (#10530)
- Release SoftTeacher checkpoints (#10119)
- Release `centernet-update_r50-caffe_fpn_ms-1x_coco` checkpoints  (#10327)
- Support SIoULoss (#10290)
- Support Eqlv2 loss (#10120)
- Support CopyPaste when mask is not available (#10509)
- Support MIM to download ODL dataset (#10460)
- Support new config (#10566)

### Bug Fixes

- Fix benchmark scripts error in windows (#10128)
- Fix error of `YOLOXModeSwitchHook` does not switch the mode when resumed from the checkpoint after switched (#10116)
- Fix pred and weight dims unmatch in SmoothL1Loss (#10423)

### Improvements

- Update MMDet_Tutorial.ipynb (#10081)
- Support to hide inference progress (#10519)
- Replace mmcls with mmpretrain  (#10545)

### Contributors

A total of 29 developers contributed to this release.

Thanks @lovelykite, @minato-ellie, @freepoet, @wufan-tb, @yalibian, @keyakiluo, @gihanjayatilaka, @i-aki-y, @xin-li-67, @RangeKing, @JingweiZhang12, @MambaWong, @lucianovk, @tall-josh, @xiuqhou, @jamiechoi1995, @YQisme, @yechenzhi, @bjzhb666, @xiexinch, @jamiechoi1995, @yarkable, @Renzhihan, @nijkah, @amaizr, @Lum1104, @zwhus, @Czm369, @hhaAndroid

## v3.0.0 (6/4/2023)

### Highlights

- Support Semi-automatic annotation Base [Label-Studio](../../../projects/LabelStudio) (#10039)
- Support [EfficientDet](../../../projects/EfficientDet) in projects (#9810)

### New Features

- File I/O migration and reconstruction (#9709)
- Release DINO Swin-L 36e model (#9927)

### Bug Fixes

- Fix benchmark script (#9865)
- Fix the crop method of PolygonMasks (#9858)
- Fix Albu augmentation with the mask shape (#9918)
- Fix `RTMDetIns` prior generator device error (#9964)
- Fix `img_shape` in data pipeline (#9966)
- Fix cityscapes import error (#9984)
- Fix `solov2_r50_fpn_ms-3x_coco.py` config error (#10030)
- Fix Conditional DETR AP and Log (#9889)
- Fix accepting an unexpected argument local-rank in PyTorch 2.0 (#10050)
- Fix `common/ms_3x_coco-instance.py` config error (#10056)
- Fix compute flops error (#10051)
- Delete `data_root` in `CocoOccludedSeparatedMetric` to fix bug (#9969)
- Unifying metafile.yml (#9849)

### Improvements

- Added BoxInst r101 config (#9967)
- Added config migration guide (#9960)
- Added more social networking links (#10021)
- Added RTMDet config introduce (#10042)
- Added visualization docs (#9938, #10058)
- Refined data_prepare docs (#9935)
- Added support for setting the cache_size_limit parameter of dynamo in PyTorch 2.0 (#10054)
- Updated coco_metric.py (#10033)
- Update type hint (#10040)

### Contributors

A total of 19 developers contributed to this release.

Thanks @IRONICBo, @vansin, @RangeKing, @Ghlerrix, @okotaku, @JosonChan1998, @zgzhengSE, @bobo0810, @yechenzh, @Zheng-LinXiao, @LYMDLUT, @yarkable, @xiejiajiannb, @chhluo, @BIGWangYuDong, @RangiLy, @zwhus, @hhaAndroid, @ZwwWayne

## v3.0.0rc6 (24/2/2023)

### Highlights

- Support [Boxinst](../../../configs/boxinst), [Objects365 Dataset](../../../configs/objects365), and [Separated and Occluded COCO metric](../user_guides/useful_tools.md#COCO-Separated-&-Occluded-Mask-Metric)
- Support [ConvNeXt-V2](../../../projects/ConvNeXt-V2), [DiffusionDet](../../../projects/DiffusionDet), and inference of [EfficientDet](../../../projects/EfficientDet) and [Detic](../../../projects/Detic) in `Projects`
- Refactor [DETR](../../../configs/detr) series and support [Conditional-DETR](../../../configs/conditional_detr), [DAB-DETR](../../../configs/dab_detr), and [DINO](../../../configs/detr)
- Support `DetInferencer` for inference, Test Time Augmentation, and automatically importing modules from registry
- Support RTMDet-Ins ONNXRuntime and TensorRT [deployment](../../../configs/rtmdet/README.md#deployment-tutorial)
- Support [calculating FLOPs of detectors](../user_guides/useful_tools.md#Model-Complexity)

### New Features

- Support [Boxinst](https://arxiv.org/abs/2012.02310) (#9525)
- Support [Objects365 Dataset](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf) (#9600)
- Support [ConvNeXt-V2](http://arxiv.org/abs/2301.00808) in `Projects` (#9619)
- Support [DiffusionDet](https://arxiv.org/abs/2211.09788) in `Projects` (#9639, #9768)
- Support [Detic](http://arxiv.org/abs/2201.02605) inference in `Projects` (#9645)
- Support [EfficientDet](https://arxiv.org/abs/1911.09070) inference in `Projects` (#9645)
- Support [Separated and Occluded COCO metric](https://arxiv.org/abs/2210.10046) (#9710)
- Support auto import modules from registry (#9143)
- Refactor DETR series and support Conditional-DETR, DAB-DETR and DINO (#9646)
- Support `DetInferencer` for inference (#9561)
- Support Test Time Augmentation (#9452)
- Support calculating FLOPs of detectors (#9777)

### Bug Fixes

- Fix deprecating old type alias due to new version of numpy (#9625, #9537)
- Fix VOC metrics (#9784)
- Fix the wrong link of RTMDet-x log (#9549)
- Fix RTMDet link in README (#9575)
- Fix MMDet get flops error (#9589)
- Fix `use_depthwise` in RTMDet (#9624)
- Fix `albumentations` augmentation post process with masks (#9551)
- Fix DETR series Unit Test (#9647)
- Fix `LoadPanopticAnnotations` bug (#9703)
- Fix `isort` CI (#9680)
- Fix amp pooling overflow (#9670)
- Fix docstring about noise in DINO (#9747)
- Fix potential bug in `MultiImageMixDataset` (#9764)

### Improvements

- Replace NumPy transpose with PyTorch permute to speed-up (#9762)
- Deprecate `sklearn` (#9725)
- Add RTMDet-Ins deployment guide (#9823)
- Update RTMDet config and README (#9603)
- Replace the models used in the tutorial document with RTMDet (#9843)
- Adjust the minimum supported python version to 3.7 (#9602)
- Support modifying palette through configuration (#9445)
- Update README document in `Project` (#9599)
- Replace `github` with `gitee` in `.pre-commit-config-zh-cn.yaml` file (#9586)
- Use official `isort` in `.pre-commit-config.yaml` file (#9701)
- Change MMCV minimum version to `2.0.0rc4` for `dev-3.x` (#9695)
- Add Chinese version of single_stage_as_rpn.md and test_results_submission.md (#9434)
- Add OpenDataLab download link (#9605, #9738)
- Add type hints of several layers (#9346)
- Add typehint for `DarknetBottleneck` (#9591)
- Add dockerfile (#9659)
- Add twitter, discord, medium, and youtube link (#9775)
- Prepare for merging refactor-detr (#9656)
- Add metafile to ConditionalDETR, DABDETR and DINO (#9715)
- Support to modify `non_blocking` parameters (#9723)
- Comment repeater visualizer register (#9740)
- Update user guide: `finetune.md` and `inference.md` (#9578)

### New Contributors

- @NoFish-528 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9346>
- @137208 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9434>
- @lyviva made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9625>
- @zwhus made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9589>
- @zylo117 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9670>
- @chg0901 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9740>
- @DanShouzhu made their first contribution in https://github.com/open-mmlab/mmdetection/pull/9578

### Contributors

A total of 27 developers contributed to this release.

Thanks @JosonChan1998, @RangeKing, @NoFish-528, @likyoo, @Xiangxu-0103, @137208, @PeterH0323, @tianleiSHI, @wufan-tb, @lyviva, @zwhus, @jshilong, @Li-Qingyun, @sanbuphy, @zylo117, @triple-Mu, @KeiChiTse, @LYMDLUT, @nijkah, @chg0901, @DanShouzhu, @zytx121, @vansin, @BIGWangYuDong, @hhaAndroid, @RangiLyu, @ZwwWayne

## v3.0.0rc5 (26/12/2022)

### Highlights

- Support [RTMDet](https://arxiv.org/abs/2212.07784) instance segmentation models. The technical report of RTMDet is on [arxiv](https://arxiv.org/abs/2212.07784)
- Support SSHContextModule in paper [SSH: Single Stage Headless Face Detector](https://arxiv.org/abs/1708.03979).

### New Features

- Support [RTMDet](https://arxiv.org/abs/2212.07784) instance segmentation models and improve RTMDet test config (#9494)
- Support SSHContextModule in paper [SSH: Single Stage Headless Face Detector](https://arxiv.org/abs/1708.03979) (#8953)
- Release [CondInst](https://arxiv.org/abs/2003.05664) pre-trained model (#9406)

### Bug Fixes

- Fix CondInst predict error when `batch_size` is greater than 1 in inference (#9400)
- Fix the bug of visualization when the dtype of the pipeline output image is not uint8 in browse dataset (#9401)
- Fix `analyze_logs.py` to plot mAP and calculate train time correctly (#9409)
- Fix backward inplace error with `PAFPN` (#9450)
- Fix config import links in model converters (#9441)
- Fix `DeformableDETRHead` object has no attribute `loss_single` (#9477)
- Fix the logic of pseudo bboxes predicted by teacher model in SemiBaseDetector (#9414)
- Fix demo API in instance segmentation tutorial (#9226)
- Fix `analyze_results` (#9380)
- Fix the error that Readthedocs API cannot be displayed (#9510)
- Fix the error when there are no prediction results and support visualize the groundtruth of TTA (#9840)

### Improvements

- Remove legacy `builder.py` (#9479)
- Make sure the pipeline argument shape is in `(width, height)` order (#9324)
- Add `.pre-commit-config-zh-cn.yaml` file (#9388)
- Refactor dataset metainfo to lowercase (#9469)
- Add PyTorch 1.13 checking in CI (#9478)
- Adjust `FocalLoss` and `QualityFocalLoss` to allow different kinds of targets (#9481)
- Refactor `setup.cfg` (#9370)
- Clip saturation value to valid range `[0, 1]` (#9391)
- Only keep meta and state_dict when publishing model (#9356)
- Add segm evaluator in ms-poly_3x_coco_instance config (#9524)
- Update deployment guide (#9527)
- Update zh_cn `faq.md` (#9396)
- Update `get_started` (#9480)
- Update the zh_cn user_guides of `useful_tools.md` and `useful_hooks.md` (#9453)
- Add type hints for `bfp` and `channel_mapper` (#9410)
- Add type hints of several losses (#9397)
- Add type hints and update docstring for task modules (#9468)

### New Contributors

- @lihua199710 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9388>
- @twmht made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9450>
- @tianleiSHI made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9453>
- @kitecats made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9481>
- @QJC123654 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9468>

### Contributors

A total of 20 developers contributed to this release.

Thanks @liuyanyi, @RangeKing, @lihua199710, @MambaWong, @sanbuphy, @Xiangxu-0103, @twmht, @JunyaoHu, @Chan-Sun, @tianleiSHI, @zytx121, @kitecats, @QJC123654, @JosonChan1998, @lvhan028, @Czm369, @BIGWangYuDong, @RangiLyu, @hhaAndroid, @ZwwWayne

## v3.0.0rc4 (23/11/2022)

### Highlights

- Support [CondInst](https://arxiv.org/abs/2003.05664)
- Add `projects/` folder, which will be a place for some experimental models/features.
- Support [SparseInst](https://arxiv.org/abs/2203.12827) in [`projects`](./projects/SparseInst/README.md)

### New Features

- Support [CondInst](https://arxiv.org/abs/2003.05664) (#9223)
- Add `projects/` folder, which will be a place for some experimental models/features (#9341)
- Support [SparseInst](https://arxiv.org/abs/2203.12827) in [`projects`](./projects/SparseInst/README.md) (#9377)

### Bug Fixes

- Fix `pixel_decoder_type` discrimination in MaskFormer Head. (#9176)
- Fix wrong padding value in cached MixUp (#9259)
- Rename `utils/typing.py` to `utils/typing_utils.py` to fix `collect_env` error (#9265)
- Fix resume arg conflict (#9287)
- Fix the configs of Faster R-CNN with caffe backbone (#9319)
- Fix torchserve and update related documentation (#9343)
- Fix bbox refine bug with sigmooid activation (#9538)

### Improvements

- Update the docs of GIoU Loss in README (#8810)
- Handle dataset wrapper in `inference_detector` (#9144)
- Update the type of `counts` in COCO's compressed RLE (#9274)
- Support saving config file in `print_config` (#9276)
- Update docs about video inference (#9305)
- Update guide about model deployment (#9344)
- Fix doc typos of useful tools (#9177)
- Allow to resume from specific checkpoint in CLI (#9284)
- Update FAQ about windows installation issues of pycocotools (#9292)

### New Contributors

- @Daa98 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9274>
- @lvhan028 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9344>

### Contributors

A total of 12 developers contributed to this release.

Thanks @sanbuphy, @Czm369, @Daa98, @jbwang1997, @BIGWangYuDong, @JosonChan1998, @lvhan028, @RunningLeon, @RangiLyu, @Daa98, @ZwwWayne, @hhaAndroid

## v3.0.0rc3 (4/11/2022)

Upgrade the minimum version requirement of MMEngine to 0.3.0 to use `ignore_key` of `ConcatDataset` for training VOC datasets (#9058)

### Highlights

- Support [CrowdDet](https://arxiv.org/abs/2003.09163) and [EIoU Loss](https://ieeexplore.ieee.org/document/9429909)
- Support training detection models in Detectron2
- Refactor Fast R-CNN

### New Features

- Support [CrowdDet](https://arxiv.org/abs/2003.09163) (#8744)
- Support training detection models in Detectron2 with examples of Mask R-CNN, Faster R-CNN, and RetinaNet (#8672)
- Support [EIoU Loss](https://ieeexplore.ieee.org/document/9429909) (#9086)

### Bug Fixes

- Fix `XMLDataset` image size error (#9216)
- Fix bugs of empty_instances when predicting without nms in roi_head (#9015)
- Fix the config file of DETR (#9158)
- Fix SOLOv2 cannot dealing with empty gt image (#9192)
- Fix inference demo (#9153)
- Add `ignore_key` in VOC `ConcatDataset` (#9058)
- Fix dumping results issue in test scripts. (#9241)
- Fix configs of training coco subsets on MMDet 3.x (#9225)
- Fix corner2hbox of HorizontalBoxes for supporting empty bboxes (#9140)

### Improvements

- Refactor Fast R-CNN (#9132)
- Clean requirements of mmcv-full due to SyncBN (#9207)
- Support training detection models in detectron2 (#8672)
- Add `box_type` support for `DynamicSoftLabelAssigner` (#9179)
- Make scipy as a default dependency in runtime (#9187)
- Update eval_metric (#9062)
- Add `seg_map_suffix` in `BaseDetDataset` (#9088)

### New Contributors

- @Wwupup made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9086>
- @sanbuphy made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9153>
- @cxiang26 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9158>
- @JosonChan1998 made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9225>

### Contributors

A total of 13 developers contributed to this release.

Thanks @wanghonglie, @Wwupup, @sanbuphy, @BIGWangYuDong, @liuyanyi, @cxiang26, @jbwang1997, @ZwwWayne, @yuyoujiang, @RangiLyu, @hhaAndroid, @JosonChan1998, @Czm369

## v3.0.0rc2 (21/10/2022)

### Highlights

- Support [imagenet pre-training](configs/rtmdet/cspnext_imagenet_pretrain) for RTMDet's backbone

### New Features

- Support [imagenet pre-training](configs/rtmdet/cspnext_imagenet_pretrain) for RTMDet's backbone (#8887)
- Add `CrowdHumanDataset` and Metric (#8430)
- Add `FixShapeResize` to support resize of fixed shape (#8665)

### Bug Fixes

- Fix `ConcatDataset` Import Error (#8909)
- Fix `CircleCI` and `readthedoc` build failed (#8980, #8963)
- Fix bitmap mask translate when `out_shape` is different (#8993)
- Fix inconsistency in `Conv2d` weight channels (#8948)
- Fix bugs when plotting loss curve by analyze_logs.py (#8944)
- Fix type change of labels in `albumentations` (#9074)
- Fix some docs and types error (#8818)
- Update memory occupation of `RTMDet` in metafile (#9098)
- Fix wrong arguments of `OpenImageMetrics` in the config (#9061)

### Improvements

- Refactor standard roi head with `box type` (#8658)
- Support mask concatenation in `BitmapMasks` and `PolygonMasks` (#9006)
- Update PyTorch and dependencies' version in dockerfile (#8845)
- Update `robustness_eval.py` and `print_config` (#8452)
- Make compatible with `ConfigDict` and `dict` in `dense_heads` (#8942)
- Support logging coco metric copypaste (#9012)
- Remove `Normalize` transform (#8913)
- Support jittering the color of different instances of the same class (#8988)
- Add assertion for missing key in `PackDetInputs` (#8982)

### New Contributors

- @Chan-Sun made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/8909>
- @MambaWong made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/8913>
- @yuyoujiang made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/8437>
- @sltlls made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/8944>
- @Nioolek made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/8845>
- @wufan-tb made their first contribution in <https://github.com/open-mmlab/mmdetection/pull/9061>

### Contributors

A total of 13 developers contributed to this release.

Thanks @RangiLyu, @jbwang1997, @wanghonglie, @Chan-Sun, @RangeKing, @chhluo, @MambaWong, @yuyoujiang, @hhaAndroid, @sltlls, @Nioolek, @ZwwWayne, @wufan-tb

## v3.0.0rc1 (26/9/2022)

### Highlights

- Release a high-precision, low-latency single-stage object detector [RTMDet](configs/rtmdet).

### Bug Fixes

- Fix UT to be compatible with PyTorch 1.6 (#8707)
- Fix `NumClassCheckHook` bug when model is wrapped (#8794)
- Update the right URL of R-50-FPN with BoundedIoULoss (#8805)
- Fix potential bug of indices in RandAugment (#8826)
- Fix some types and links (#8839, #8820, #8793, #8868)
- Fix incorrect background fill values in `FSAF` and `RepPoints` Head (#8813)

### Improvements

- Refactored anchor head and base head with `box type` (#8625)
- Refactored `SemiBaseDetector` and `SoftTeacher` (#8786)
- Add list to dict keys to avoid modify loss dict (#8828)
- Update `analyze_results.py` , `analyze_logs.py` and `loading.py` (#8430, #8402, #8784)
- Support dump results in `test.py` (#8814)
- Check empty predictions in `DetLocalVisualizer._draw_instances` (#8830)
- Fix `floordiv` warning in `SOLO` (#8738)

### Contributors

A total of 16 developers contributed to this release.

Thanks @ZwwWayne, @jbwang1997, @Czm369, @ice-tong, @Zheng-LinXiao, @chhluo, @RangiLyu, @liuyanyi, @wanghonglie, @levan92, @JiayuXu0, @nye0, @hhaAndroid, @xin-li-67, @shuxp, @zytx121

## v3.0.0rc0 (31/8/2022)

We are excited to announce the release of MMDetection 3.0.0rc0. MMDet 3.0.0rc0 is the first version of MMDetection 3.x, a part of the OpenMMLab 2.0 projects. Built upon the new [training engine](https://github.com/open-mmlab/mmengine), MMDet 3.x unifies the interfaces of the dataset, models, evaluation, and visualization with faster training and testing speed. It also provides a general semi-supervised object detection framework and strong baselines.

### Highlights

1. **New engine**. MMDet 3.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a universal and powerful runner that allows more flexible customizations and significantly simplifies the entry points of high-level interfaces.

2. **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMDet 3.x unifies and refactors the interfaces and internal logic of training, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logic to allow the emergence of multi-task/modality algorithms.

3. **Faster speed**. We optimize the training and inference speed for common models and configurations, achieving a faster or similar speed than [Detection2](https://github.com/facebookresearch/detectron2/). Model details of benchmark will be updated in [this note](./benchmark.md#comparison-with-detectron2).

4. **General semi-supervised object detection**. Benefitting from the unified interfaces, we support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.

5. **Strong baselines**. We release strong baselines of many popular models to enable fair comparisons among state-of-the-art models.

6. **New features and algorithms**:

   - Enable all the single-stage detectors to serve as region proposal networks
   - [SoftTeacher](https://arxiv.org/abs/2106.09018)
   - [the updated CenterNet](https://arxiv.org/abs/2103.07461)

7. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmdetection.readthedocs.io/en/3.x/).

### Breaking Changes

MMDet 3.x has undergone significant changes for better design, higher efficiency, more flexibility, and more unified interfaces.
Besides the changes in API, we briefly list the major breaking changes in this section.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.
Users can also refer to the [API doc](https://mmdetection.readthedocs.io/en/3.x/) for more details.

#### Dependencies

- MMDet 3.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMDet 3.x is not guaranteed.
- MMDet 3.x relies on MMEngine to run. MMEngine is a new foundational library for training deep learning models of OpenMMLab and is the core dependency of OpenMMLab 2.0 projects. The dependencies of file IO and training are migrated from MMCV 1.x to MMEngine.
- MMDet 3.x relies on MMCV>=2.0.0rc0. Although MMCV no longer maintains the training functionalities since 2.0.0rc0, MMDet 3.x relies on the data transforms, CUDA operators, and image processing interfaces in MMCV. Note that the package `mmcv` is the version that provides pre-built CUDA operators and `mmcv-lite` does not since MMCV 2.0.0rc0, while `mmcv-full` has been deprecated since 2.0.0rc0.

#### Training and testing

- MMDet 3.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of the dataset, model, evaluation, and visualizer. Therefore, MMDet 3.x no longer maintains the building logic of those modules in `mmdet.train.apis` and `tools/train.py`. Those codes have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic to that in training scripts to build the runner.
- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.
- Learning rate and momentum schedules have been migrated from Hook to [Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structure to ease the understanding of the components in the runner. Users can read the [config example of MMDet 3.x](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. The names of checkpoints are not updated for now as there is no BC-breaking of model weights between MMDet 3.x and 2.x. We will progressively replace all the model weights with those trained in MMDet 3.x. Please refer to the [user guides of config](../user_guides/config.md) for more details.

#### Dataset

The Dataset classes implemented in MMDet 3.x all inherit from the `BaseDetDataset`, which inherits from the [BaseDataset in MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). In addition to the changes in interfaces, there are several changes in Dataset in MMDet 3.x.

- All the datasets support serializing the internal data list to reduce the memory when multiple workers are built for data loading.
- The internal data structure in the dataset is changed to be self-contained (without losing information like class names in MMDet 2.x) while keeping simplicity.
- The evaluation functionality of each dataset has been removed from the dataset so that some specific evaluation metrics like COCO AP can be used to evaluate the prediction on other datasets.

#### Data Transforms

The data transforms in MMDet 3.x all inherits from `BaseTransform` in MMCV>=2.0.0rc0, which defines a new convention in OpenMMLab 2.0 projects.
Besides the interface changes, there are several changes listed below:

- The functionality of some data transforms (e.g., `Resize`) are decomposed into several transforms to simplify and clarify the usages.
- The format of data dict processed by each data transform is changed according to the new data structure of dataset.
- Some inefficient data transforms (e.g., normalization and padding) are moved into data preprocessor of model to improve data loading and training speed.
- The same data transforms in different OpenMMLab 2.0 libraries have the same augmentation implementation and the logic given the same arguments, i.e., `Resize` in MMDet 3.x and MMSeg 1.x will resize the image in the exact same manner given the same arguments.

#### Model

The models in MMDet 3.x all inherit from `BaseModel` in MMEngine, which defines a new convention of models in OpenMMLab 2.0 projects.
Users can refer to [the tutorial of the model in MMengine](https://mmengine.readthedocs.io/en/latest/tutorials/model.html) for more details.
Accordingly, there are several changes as the following:

- The model interfaces, including the input and output formats, are significantly simplified and unified following the new convention in MMDet 3.x.
  Specifically, all the input data in training and testing are packed into `inputs` and `data_samples`, where `inputs` contains model inputs like a list of image tensors, and `data_samples` contains other information of the current data sample such as ground truths, region proposals, and model predictions. In this way, different tasks in MMDet 3.x can share the same input arguments, which makes the models more general and suitable for multi-task learning and some flexible training paradigms like semi-supervised learning.
- The model has a data preprocessor module, which is used to pre-process the input data of the model. In MMDet 3.x, the data preprocessor usually does the necessary steps to form the input images into a batch, such as padding. It can also serve as a place for some special data augmentations or more efficient data transformations like normalization.
- The internal logic of the model has been changed. In MMdet 2.x, model uses `forward_train`, `forward_test`, `simple_test`, and `aug_test` to deal with different model forward logics. In MMDet 3.x and OpenMMLab 2.0, the forward function has three modes: 'loss', 'predict', and 'tensor' for training, inference, and tracing or other purposes, respectively.
  The forward function calls `self.loss`, `self.predict`, and `self._forward` given the modes 'loss', 'predict', and 'tensor', respectively.

#### Evaluation

The evaluation in MMDet 2.x strictly binds with the dataset. In contrast, MMDet 3.x decomposes the evaluation from dataset so that all the detection datasets can evaluate with COCO AP and other metrics implemented in MMDet 3.x.
MMDet 3.x mainly implements corresponding metrics for each dataset, which are manipulated by [Evaluator](https://mmengine.readthedocs.io/en/latest/design/evaluator.html) to complete the evaluation.
Users can build an evaluator in MMDet 3.x to conduct offline evaluation, i.e., evaluate predictions that may not produce in MMDet 3.x with the dataset as long as the dataset and the prediction follow the dataset conventions. More details can be found in the [tutorial in mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html).

#### Visualization

The functions of visualization in MMDet 2.x are removed. Instead, in OpenMMLab 2.0 projects, we use [Visualizer](https://mmengine.readthedocs.io/en/latest/design/visualization.html) to visualize data. MMDet 3.x implements `DetLocalVisualizer` to allow visualization of ground truths, model predictions, feature maps, etc., at any place. It also supports sending the visualization data to any external visualization backends such as Tensorboard.

### Improvements

- Optimized training and testing speed of FCOS, RetinaNet, Faster R-CNN, Mask R-CNN, and Cascade R-CNN. The training speed of those models with some common training strategies is also optimized, including those with synchronized batch normalization and mixed precision training.
- Support mixed precision training of all the models. However, some models may get undesirable performance due to some numerical issues. We will update the documentation and list the results (accuracy of failure) of mixed precision training.
- Release strong baselines of some popular object detectors. Their accuracy and pre-trained checkpoints will be released.

### Bug Fixes

- DeepFashion dataset: the config and results have been updated.

### New Features

1. Support a general semi-supervised learning framework that works with all the object detectors supported in MMDet 3.x. Please refer to [semi-supervised object detection](../user_guides/semi_det.md) for details.
2. Enable all the single-stage detectors to serve as region proposal networks. We give [an example of using FCOS as RPN](../user_guides/single_stage_as_rpn.md).
3. Support a semi-supervised object detection algorithm: [SoftTeacher](https://arxiv.org/abs/2106.09018).
4. Support [the updated CenterNet](https://arxiv.org/abs/2103.07461).
5. Support data structures `HorizontalBoxes` and `BaseBoxes` to encapsulate different kinds of bounding boxes. We are migrating to use data structures of boxes to replace the use of pure tensor boxes. This will unify the usages of different kinds of bounding boxes in MMDet 3.x and MMRotate 1.x to simplify the implementation and reduce redundant codes.

### Planned changes

We list several planned changes of MMDet 3.0.0rc0 so that the community could more comprehensively know the progress of MMDet 3.x. Feel free to create a PR, issue, or discussion if you are interested, have any suggestions and feedback, or want to participate.

1. Test-time augmentation: which is supported in MMDet 2.x, is not implemented in this version due to the limited time slot. We will support it in the following releases with a new and simplified design.
2. Inference interfaces: unified inference interfaces will be supported in the future to ease the use of released models.
3. Interfaces of useful tools that can be used in Jupyter Notebook or Colab: more useful tools that are implemented in the `tools` directory will have their python interfaces so that they can be used in Jupyter Notebook, Colab, and downstream libraries.
4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMDet 3.x.
5. Wandb visualization: MMDet 2.x supports data visualization since v2.25.0, which has not been migrated to MMDet 3.x for now. Since WandB provides strong visualization and experiment management capabilities, a `DetWandBVisualizer` and maybe a hook are planned to fully migrate those functionalities from MMDet 2.x.
6. Full support of WiderFace dataset (#8508) and Fast R-CNN: we are verifying their functionalities and will fix related issues soon.
7. Migrate DETR-series algorithms (#8655, #8533) and YOLOv3 on IPU (#8552) from MMDet 2.x.

### Contributors

A total of 11 developers contributed to this release.
Thanks @shuxp, @wanghonglie, @Czm369, @BIGWangYuDong, @zytx121, @jbwang1997, @chhluo, @jshilong, @RangiLyu, @hhaAndroid, @ZwwWayne

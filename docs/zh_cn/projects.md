# 基于 MMDetection 的项目

有许多开源项目都是基于 MMDetection 搭建的，我们在这里列举一部分作为样例，展示如何基于 MMDetection 搭建您自己的项目。
由于这个页面列举的项目并不完全，我们欢迎社区提交 Pull Request 来更新这个文档。

## MMDetection 的拓展项目

一些项目拓展了 MMDetection 的边界，如将 MMDetection 拓展支持 3D 检测或者将 MMDetection 用于部署。
它们展示了 MMDetection 的许多可能性，所以我们在这里也列举一些。

- [OTEDetection](https://github.com/opencv/mmdetection): OpenVINO training extensions for object detection.
- [MMDetection3d](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.

## 研究项目

同样有许多研究论文是基于 MMDetection 进行的。许多论文都发表在了顶级的会议或期刊上，或者对社区产生了深远的影响。
为了向社区提供一个可以参考的论文列表，帮助大家开发或者比较新的前沿算法，我们在这里也遵循会议的时间顺序列举了一些论文。
MMDetection 中已经支持的算法不在此列。

- Involution: Inverting the Inherence of Convolution for Visual Recognition, CVPR21. [[paper]](https://arxiv.org/abs/2103.06255)[[github]](https://github.com/d-li14/involution)
- Multiple Instance Active Learning for Object Detection, CVPR 2021. [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yuan_Multiple_Instance_Active_Learning_for_Object_Detection_CVPR_2021_paper.pdf)[[github]](https://github.com/yuantn/MI-AOD)
- Adaptive Class Suppression Loss for Long-Tail Object Detection, CVPR 2021. [[paper]](https://arxiv.org/abs/2104.00885)[[github]](https://github.com/CASIA-IVA-Lab/ACSL)
- Generalizable Pedestrian Detection: The Elephant In The Room, CVPR2021. [[paper]](https://arxiv.org/abs/2003.08799)[[github]](https://github.com/hasanirtiza/Pedestron)
- Group Fisher Pruning for Practical Network Compression, ICML2021. [[paper]](https://github.com/jshilong/FisherPruning/blob/main/resources/paper.pdf)[[github]](https://github.com/jshilong/FisherPruning)
- Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax, CVPR2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf)[[github]](https://github.com/FishYuLi/BalancedGroupSoftmax)
- Coherent Reconstruction of Multiple Humans from a Single Image, CVPR2020. [[paper]](https://jiangwenpl.github.io/multiperson/)[[github]](https://github.com/JiangWenPL/multiperson)
- Look-into-Object: Self-supervised Structure Modeling for Object Recognition, CVPR 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Look-Into-Object_Self-Supervised_Structure_Modeling_for_Object_Recognition_CVPR_2020_paper.pdf)[[github]](https://github.com/JDAI-CV/LIO)
- Video Panoptic Segmentation, CVPR2020. [[paper]](https://arxiv.org/abs/2006.11339)[[github]](https://github.com/mcahny/vps)
- D2Det: Towards High Quality Object Detection and Instance Segmentation, CVPR2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.html)[[github]](https://github.com/JialeCao001/D2Det)
- CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection, CVPR2020. [[paper]](https://arxiv.org/abs/2003.09119)[[github]](https://github.com/KiveeDong/CentripetalNet)
- Learning a Unified Sample Weighting Network for Object Detection, CVPR 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Cai_Learning_a_Unified_Sample_Weighting_Network_for_Object_Detection_CVPR_2020_paper.html)[[github]](https://github.com/caiqi/sample-weighting-network)
- Scale-equalizing Pyramid Convolution for Object Detection, CVPR2020. [[paper]](https://arxiv.org/abs/2005.03101) [[github]](https://github.com/jshilong/SEPC)
- Revisiting the Sibling Head in Object Detector, CVPR2020. [[paper]](https://arxiv.org/abs/2003.07540)[[github]](https://github.com/Sense-X/TSD)
- PolarMask: Single Shot Instance Segmentation with Polar Representation, CVPR2020. [[paper]](https://arxiv.org/abs/1909.13226)[[github]](https://github.com/xieenze/PolarMask)
- Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection, CVPR2020. [[paper]](https://arxiv.org/abs/2003.11818)[[github]](https://github.com/ggjy/HitDet.pytorch)
- ZeroQ: A Novel Zero Shot Quantization Framework, CVPR2020. [[paper]](https://arxiv.org/abs/2001.00281)[[github]](https://github.com/amirgholami/ZeroQ)
- CBNet: A Novel Composite Backbone Network Architecture for Object Detection, AAAI2020. [[paper]](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuY.1833.pdf)[[github]](https://github.com/VDIGPKU/CBNet)
- RDSNet: A New Deep Architecture for Reciprocal Object Detection and Instance Segmentation, AAAI2020. [[paper]](https://arxiv.org/abs/1912.05070)[[github]](https://github.com/wangsr126/RDSNet)
- Training-Time-Friendly Network for Real-Time Object Detection, AAAI2020. [[paper]](https://arxiv.org/abs/1909.00700)[[github]](https://github.com/ZJULearning/ttfnet)
- Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution, NeurIPS 2019. [[paper]](https://arxiv.org/abs/1909.06720)[[github]](https://github.com/thangvubk/Cascade-RPN)
- Reasoning R-CNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection, CVPR2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)[[github]](https://github.com/chanyn/Reasoning-RCNN)
- Learning RoI Transformer for Oriented Object Detection in Aerial Images, CVPR2019. [[paper]](https://arxiv.org/abs/1812.00155)[[github]](https://github.com/dingjiansw101/AerialDetection)
- SOLO: Segmenting Objects by Locations. [[paper]](https://arxiv.org/abs/1912.04488)[[github]](https://github.com/WXinlong/SOLO)
- SOLOv2: Dynamic, Faster and Stronger. [[paper]](https://arxiv.org/abs/2003.10152)[[github]](https://github.com/WXinlong/SOLO)
- Dense Peppoints: Representing Visual Objects with Dense Point Sets. [[paper]](https://arxiv.org/abs/1912.11473)[[github]](https://github.com/justimyhxu/Dense-RepPoints)
- IterDet: Iterative Scheme for Object Detection in Crowded Environments. [[paper]](https://arxiv.org/abs/2005.05708)[[github]](https://github.com/saic-vul/iterdet)
- Cross-Iteration Batch Normalization. [[paper]](https://arxiv.org/abs/2002.05712)[[github]](https://github.com/Howal/Cross-iterationBatchNorm)
- A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection, NeurIPS2020 [[paper]](https://arxiv.org/abs/2009.13592)[[github]](https://github.com/kemaloksuz/aLRPLoss)

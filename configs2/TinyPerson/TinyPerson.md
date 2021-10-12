# TinyPerson

- [Scale Match Experiment](scale_match/ScaleMatch.md)
- [Coarse Point Experiment](coarsepoint/CoarsePoint.md)

## fixed
- there are two times which can merge results of sub images: 'during inference' or 'after inference', 
last version we use 'during inference' policy and keep max_per_img=100, but an full image can have 800+ person.
So the right setting is max_per_img=200 for 'after inference' policy, or max_per_img=1000 for 'during inference' policy

### 1 配置文件
配置dataset和mini_annotations

config相关文件的添加,修改涉及
- _base_里的dataset
- num_class/max_per_img && nms_pre
- anchor scales
- fix BN requires_grad=False in Backbone (learnable BN/GN is hard for TinyPerson??)
- adap: neck.start_idx and anchor strides 

```
# dataset
configs2/_base_/datasets/TinyPerson/TinyPerson_detection_640x512.py

# Faster-FPN
configs2\TinyPerson\base\faster_rcnn_r50_fpn_1x_TinyPerson640.py

# RetinaNet
configs2/TinyPerson/base/retinanet_r50_fpn_1x_TinyPerson640.py
configs2/TinyPerson/base/retinanet_r50_fpns4_1x_TinyPerson640.py
configs2/TinyPerson/base/retinanet_r50_fpns4_1x_TinyPerson640_clipg.py
```

### 2. performance

All train and test on 2080Ti, 
- CUDA10.1/10.2
- python3.7, cudatookit=10.2, pytorch=1.5, torchvision=0.6

for Faster-FPN, we think the gain compare to TinyBenchmark may come 
from the cut and merge during inference running time. 

detector | num_gpu | $AP_{50}^{tiny}$| script
--- | --- | ---| ---
Faster-FPN | 4 | 49.81(1) | base/Baseline_TinyPerson.sh:exp1.1
Adap RetainaNet | 1 | 45.85(1) | base/Baseline_TinyPerson.sh:exp2.1
Adap RetainaNet | 4 | 46.52(1) | base/Baseline_TinyPerson.sh:exp2.2(clip grad)
Adap FCOS | 2 | 47.61(1) | base/Baseline_TinyPerson.sh:exp6.1

#### test time compare

- base/Baseline_TinyPerson.sh:exp5.1
- detector: Adap FCOS

run-time crop | nms_pre | max_per_img | max_det|$AP_{50}^{tiny}$
 --- | --- | ---| --- | ---
 Y | 1000 | 100  | 200 | 42.93
 Y | 5000 | 1000 | 200 | 46.11
 Y | 2000 | 1000 | 200 | 46.11
 Y | 1000 | 1000 | 200 | 46.11
 Y | 2000 | 1000 | 1000 | 47.61
 N | 1000/crop | 100/crop | 200 | 45.68
 
run-time crop | nms_pre | max_per_img | max_det|$AP_{50}^{tiny}$
 --- | --- | ---| --- | ---
 Y | 2000 | 1000 | 1000 | 47.61
 N | 1000/crop | 500/crop | 1000| 46.86
 N | 1000/crop | 100/crop | 200 | 45.68
 
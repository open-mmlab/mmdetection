# PCB

```shell
git clone https://github.com/open-mmlab/mmdetection -b dev-3.x
cd mmdetection

mkdir data
cd data
wget https://openmmlab.vansin.top/datasets/pcb.zip
unzip pcb.zip -d pcb
```

browse datasets

```shell
python tools/analysis_tools/browse_dataset.py projects/application/pcb/rtmdet_l-300e_coco.py --output-dir output
```

![image](https://user-images.githubusercontent.com/60632596/226512901-ec447734-8f51-4a94-a52a-ece162cf1ee2.png)

train

```shell
python tools/train.py projects/application/pcb/rtmdet_l-300e_coco.py
```

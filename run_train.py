import sys
import subprocess

args = ['python']
args.append('tools/train.py')
args.append('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
subprocess.run(args)


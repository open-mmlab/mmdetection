import numpy as np
import json
import os



def add_iscrowd(json_path, save_path):
    with open(json_path, 'r') as f:
        ann = json.load(f)
    
    for ix in range(len(ann['annotations'])):
        ann['annotations'][ix]['iscrowd'] = 0
    
    with open(save_path, 'w') as f:
        json.dump(ann, f)


if __name__=='__main__':
    json_old_path = '/SSDb/sung/dataset/seadrone/annotations/instances_old_val.json'
    json_new_path = '/SSDb/sung/dataset/seadrone/annotations/instances_val.json'
    add_iscrowd(json_old_path, json_new_path)
import json
from glob import glob
import os
import pandas as pd


def rule1(name):
    return '-'.join(name.split('-')[-2:])

def report(dir, rule):
    json_file = glob(os.path.join(dir, '*.json'))
    if len(json_file) > 1:
        return [rule(dir), '', '', '', '', '']
    else:
        json_file = json_file[0]
        result = []
        with open(json_file, 'r') as f:
            for line in f:
                result.append(json.loads(line))
        
        result_output = [rule(dir), result[-1]['epoch'], \
            result[-1]['bbox_mAP'], result[-1]['bbox_mAP_50'], \
            result[-1]['bbox_mAP_75'], result[-1]["bbox_mAP_s"], \
            result[-1]["bbox_mAP_m"], result[-1]["bbox_mAP_l"]
        ]
        return result_output


if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Wrapup
    column = ['Name', 'Epoch', 'mAP', 'mAP@50', 'mAP@75', 'mAP_S', 'mAP-M', 'mAP-L']
    final_output = []
    dir_list = glob('../result/test/coco/*')
    
    for dir in dir_list:
        # Single Dir Extraction
        output = report(dir, rule1)
        final_output.append(output)
    
    result = pd.DataFrame(final_output, columns=column)    
    result.to_csv('hyperparameter_tuning.csv', index=False)
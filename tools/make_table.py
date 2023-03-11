import os
import json
import pandas as pd

root_path = '/ailab_mat/checkpoints/sung/msdet/output/'
folder_list = [_ for _ in os.listdir(root_path) if '.zip' not in _]
folder_list.sort()
print(len(folder_list))
df = pd.DataFrame(columns=["model", "AP@50:5:95", "AP@50", "AP@75", "AP@S", "AP@M", "AP@L"])

for i, folder in enumerate(folder_list):
    path = root_path + folder + '/scores.txt'
    with open(path, 'r') as f:
        scores = f.read()
    # print(scores)
    model = folder.split('_')[2]
    if 'fr' in model:
        model_name = 'faster_rcnn'
    elif 'fc' in model:
        model_name = 'fcos'
    elif 'mr' in model:
        model_name = 'mask_rcnn'
    elif 'sr' in model:
        model_name = 'sparse_rcnn'
    model = model[2:]
    
    if '50' in model:
        model_name += '_r50_'
        model = model[3:]
    elif 'x101' in model:
        model_name += '_x101_'
        model = model[5:]
    elif '101' in model:
        model_name += '_r101_'
        model = model[4:]

    

    df.loc[i] = [model_name + model, scores[4:9], scores[17:22], scores[30:35], scores[46:51], scores[63:68], scores[79:84]]
    # print()

print(df)

df.to_csv('table.csv', sep=',', index=False)
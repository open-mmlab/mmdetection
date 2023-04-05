import pickle
import os

file_path = os.path.join('work_dirs/faster_rcnn_r50_fpn_2x_HRSID/results.pkl')
f = open(file_path, 'rb')
data = pickle.load(f)
print(data)
import os


models = os.listdir('/ailab_mat/checkpoints/sung/msdet/coco_teacher')
# models = os.listdir('/ailab_mat/checkpoints/sung/msdet/coco_student')
MODEL_NAME = []
FOLDER_NAME = []
EPOCH_NUMBER = []

for m in models:
    model = m.split('_')[0] + '_' + m.split('_')[1]
    f = model + '_kd'
    if '1x' in m:
        e = str(12)
    elif '2x' in m:
        e = str(24)
    elif '3x' in m:
        e = str(36)
    else:
        print('error: epoch')
    MODEL_NAME.append(m)
    FOLDER_NAME.append(f)
    EPOCH_NUMBER.append(e)
    print(m, '\t\t', f,'\t\t', e)
print('########')
print(MODEL_NAME)
print(FOLDER_NAME)
print(EPOCH_NUMBER)

print(len(MODEL_NAME))
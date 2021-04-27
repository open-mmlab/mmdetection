from mmdet.apis import init_detector, inference_detector
import os
import numpy as np

config_file = '../configs/tinaface/tinaface_r50_fpn_1x_widerface.py'

checkpoint_file = '../work_dirs/tinaface_r50_fpn_widerface.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

#eval_root = '../dataset/WIDER_test/images'
eval_root = '../data/WIDERFace/WIDER_val'
eval_list = os.listdir(eval_root)
predict_root = '/disk1/mh/projects/test/mmdetection/Evaluate/WFPred'

for dire in eval_list:
    if dire == 'Annotations':
            continue
    
    if os.path.exists(os.path.join(predict_root, dire)):
        pass
    else:
        os.mkdir(os.path.join(predict_root, dire))
        #pass
    
    image_folder_path = os.path.join(eval_root, dire)
    image_class_list = os.listdir(image_folder_path)
    
    class_save_path = os.path.join(predict_root, dire)
        
    if os.path.exists(class_save_path):
        pass
    else:
        os.mkdir(class_save_path)
    
    #print(image_folder_path)
    for image_name in image_class_list:
        
        ############data/WIDERFace/WIDER_val/9--Press_Conference/9_Press_Conference_Press_Conference_9_945.jpg
        image_file_path = os.path.join(image_folder_path, image_name)
        #print(image_file_path)
        
        result = inference_detector(model, image_file_path)
        
        #location=result[0][result[0][: , -1] > 0.3][: , 0:-1].astype(int).astype(str)
        location=result[0][result[0][: , -1] > 0][: , 0:-1].astype(int)
        location[: , 2] =  location[: , 2] - location[: , 0]
        location[: , 3] =  location[: , 3] - location[: , 1]
        location=location.astype(str)
        confidence= result[0][result[0][: , -1] > 0][: ,-1].astype(str)
        confidence = confidence[:,np.newaxis]
        value = np.hstack((location,confidence))
        #print(value)
        
        
        image_save_path = os.path.join(class_save_path, image_name.replace('.jpg','.txt'))
        print(image_save_path)
        
        np.savetxt(image_save_path ,value,fmt = '%s')
        fp = open(image_save_path)
        lines = []
        for line in fp: 
            lines.append(line[0:-1])
        fp.close()

        lines.insert(0, str(len(result[0][result[0][: , -1] > 0])))
        lines.insert(0, image_name.replace('.jpg','')) 
        s = '\n'.join(lines)
        fp = open(image_save_path, 'w')
        fp.write(s)
        fp.close()
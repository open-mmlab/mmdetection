import os
import shutil

origin_VOC = "/workspace/data/MM_det/data/Pascal/VOCdevkit/"
output_VOC = "/workspace/data/MM_det/data/Pascal/PascalVOC/"

'''
    -VOC
        -annotations
        -2007
        -2012
'''
# for year in os.listdir(origin_VOC):
#     y_root = os.path.join(origin_VOC, year)
#     with open(y_root+"/ImageSets/Main/trainval.txt", 'r') as trainval:
#         for line in trainval.readlines():
#             line = line.strip()
#             shutil.copy(y_root+"/JPEGImages/"+line+'.jpg', \
#                 output_VOC+"train0712/"+line+'.jpg')

# y_root = os.path.join(origin_VOC, "VOC2007")
# with open(y_root+"/ImageSets/Main/test.txt", 'r') as test:
#     for line in test.readlines():
#         line = line.strip()
#         shutil.copy(y_root+"/JPEGImages/"+line+'.jpg', \
#             output_VOC+"test07/"+line+'.jpg')

import time
import cv2
import numpy as np
import os
import sys

def convert_coord_from_txt(filename):
    start_frame = 0

    with open(filename, "r") as filestream:
      with open("det.txt", "w") as filestreamtwo:
        for line in filestream:
            currentline = line.split(",")
            frame_num = int(currentline[0])

            bb_width = float(currentline[3]) - float(currentline[1])
            bb_height = float(currentline[4])-float(currentline[2])

            x1 = int(float(currentline[1]))
            y1 = int(float(currentline[2]))
            x2 = int(float(currentline[3])) 
            y2 = int(float(currentline[4]))
            
            total = str(int(float(currentline[0]))-start_frame) + "," + "-1" + "," + str(int(float(currentline[1]))) + "," + str(int(float(currentline[2]))) + "," + str(bb_width) + "," + str(bb_height) + "," + str(1.0) + "," + "-1" + "," + "-1" + "," + "-1" + "\n"
            filestreamtwo.write(total)

if __name__ == '__main__':
    convert_coord_from_txt(sys.argv[1])

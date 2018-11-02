# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:39:02 2018

@author: telos
"""

import cv2
import os

FILE_NAME = 'video.mp4' #이름 변경 가능 
FOLDER_NAME = 'imagefolder'
FRAME_COUNT = 10 #몇 프레임에 한번 저장할지 결정합니다. 디폴트는 10입니다.

cap = cv2.VideoCapture(FILE_NAME)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

try:
    os.mkdir(FOLDER_NAME)
    index = 0
except:
    file_list = os.listdir(FOLDER_NAME)
    index = len(file_list)

count = 0
while True:
    retval, frame = cap.read()
    if not retval:
        break
    if count % FRAME_COUNT == 0:
        img_name = FOLDER_NAME + '/%d'%index
        cv2.imwrite(img_name + '.jpg', frame) 
        index += 1
        count += 1
    count += 1
    
cap.release()
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:24:37 2018

@author: telos
"""

import cv2

VIDEO_NAME = 'video7' #이름 변경 가

cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_NAME + '.mp4', fourcc, 30.0, frame_size)

while True:
    retval, frame = cap.read()
    if not retval:
        break
    out.write(frame)
    cv2.imshow('video', frame)
    key = cv2.waitKey(25)
    if key == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 09:26:47 2018

@author: telos
"""

from keras.models import load_model
import pickle
import os
import cv2

model_name = 'cam_trained_model.h5' #저장한 케라스 모델의 파일명을 써줍니다.
model = load_model('./models/' + model_name)
SIZE = (64, 64)  #처음 선언한 사이즈를 적습니다.

def unpickle(file):
    with open(os.path.join(os.getcwd(), file), 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

#캠 데이터
data = unpickle('pic.bin')
mapping = {i:data['label_name'][i] for i in range(len(data['label_name']))}

#영상을 실시간으로 줄 경우
cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 20)
while True:
    retval, frame = cap.read()
    if not retval:
        break
    img = cv2.resize(frame, dsize = SIZE)
    size = [1, SIZE[0], SIZE[1], 3]
    img = img.reshape(size)
    r = model.predict(img)
    a = r.argmax()
    text = mapping[a]
    cv2.putText(frame, text, org, font, 1, (255, 0, 0), 2)
    cv2.imshow('video', frame)
    key = cv2.waitKey(25)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
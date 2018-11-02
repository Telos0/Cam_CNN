# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:24:57 2018

@author: telos
"""
from keras.models import load_model
import pickle
import os
import cv2

model_name = 'cam_trained_model.h5' #저장한 케라스 모델의 파일명을 써줍니다.
model = load_model('./models/' + model_name)
SIZE = (64, 64)  #처음 선언한 사이즈를 적습니다.
FILE_NAME = '11.jpg' #확인할 사진의 파일명을 넣어줍니다. 절대경로나 혹은 해당폴더의 상대경로를 넣어주십시오.

def unpickle(file):
    with open(os.path.join(os.getcwd(), file), 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

#캠 데이터
data = unpickle('pic.bin')
mapping = {i:data['label_name'][i] for i in range(len(data['label_name']))}

#먼저 사진을 확인합니다. 키보드의 아무키나 누르면 어떤 사진인지 리턴합니다.
img = cv2.imread(FILE_NAME) 
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
img = cv2.resize(img, dsize = SIZE)
size = [1, SIZE[0], SIZE[1], 3]
img = img.reshape(size) #SIZE를 중간에 넣어줍니다. 
r = model.predict(img)
a = r.argmax()
print('사진의 물체는 : ', end = '')
print(mapping[a])


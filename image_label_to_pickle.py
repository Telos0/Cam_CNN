# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:11:11 2018

@author: telos
"""

import random
import pickle
import cv2
import os
import collections
import numpy as np
from sklearn.model_selection import train_test_split

FOLDER_NAME = 'imagefolder'
SIZE = (64, 64)  #사이즈 변경 가능

file_list = os.listdir(FOLDER_NAME)
names = [i for i in range(len(file_list))]
random.shuffle(names)
image_label = collections.OrderedDict()
image_label['data'] = []
image_label['label'] = []
image_label['label_name'] = []

print('먼저 라벨을 만듭니다')
CLASS_NUM = int(input('클래스 개수를 넣어주세요: '))
for i in range(CLASS_NUM):
    value = input('해당 인덱스의 클래스 이름을 넣어주세요 index = %d : '%i)
    image_label['label_name'].append(value)

#순차적으로 섞인 사진을 보여줍니다. 라벨을 붙이지 않을것이라면 esc를 눌러주세요. 라벨을 붙일 경우 esc를 제외한
#아무 키보드나 누른후 사진에 붙일 라벨 번호를 넣어주세요.
index = 1
for image in file_list:
    img = cv2.imread(FOLDER_NAME + '/' + image)
    cv2.imshow('%d / %d'%(index, len(file_list)), img)
    key = cv2.waitKey()
    if key == 27:
        print('라벨을 붙이지 않습니다.')
        cv2.destroyAllWindows()
        index += 1
        continue
    else:
        cv2.destroyAllWindows()
        index += 1
        value = input('라벨을 넣어주세요 0~%d : '%(CLASS_NUM - 1))
        value = int(value)
        if value < 0 or value > CLASS_NUM :
            print('잘못된 입력입니다')
            break
        img = cv2.resize(img, dsize = SIZE)
        image_label['data'].append(img)
        image_label['label'].append(value)

X_train, X_test, y_train, y_test = train_test_split(image_label['data'], image_label['label'], test_size = 0.3)
data = collections.OrderedDict()
data['X_train'] = np.array(X_train)
data['X_test'] = np.array(X_test)
data['y_train'] = np.array(y_train)
data['y_test'] = np.array(y_test)
data['label_name'] = image_label['label_name']

#data: 넘파이 어레이 label: 라벨 label_name: 클래스 이름 
with open('pic.bin', 'wb') as file:
    pickle.dump(data, file)
    

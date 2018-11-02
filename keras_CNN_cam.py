# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:35:58 2018

@author: telos
"""

import keras
from keras.models import Sequential
import pickle
import os
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

batch_size = 10     #변경가능
epochs = 15        #변경가능
save_dir = 'models'
model_name = 'cam_trained_model.h5'

def unpickle(file):
    with open(os.path.join(os.getcwd(), file), 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

data = unpickle('pic.bin')
num_classes = len(data['label_name'])
x_train, y_train, x_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

camCNN = Sequential()
camCNN.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=x_train.shape[1:]))
camCNN.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
camCNN.add(MaxPooling2D(pool_size=(2, 2)))
camCNN.add(Dropout(0.2))
camCNN.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
camCNN.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
camCNN.add(MaxPooling2D(pool_size=(2, 2)))
camCNN.add(Dropout(0.3))
camCNN.add(Flatten())
camCNN.add(Dense(512, activation='relu'))
camCNN.add(Dropout(0.5))
camCNN.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
camCNN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#정규화 
x_train = x_train / 255.0
x_test = x_test / 255.0

camCNN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
camCNN.save(model_path)
# -*- coding: utf-8 -*-
"""

@author: Abhilash Srivastava
"""
from __future__ import print_function
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Flatten,Dropout,Dense,Activation
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD,adam
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json
#for reproducing the random values
np.random.seed(1671)

#network setup
EPOCH=25
BATCH_SIZE=128
VERBOSE=1
#number of outputs,10 for 10 digits
CLASSES=10
#stochastic gradient descent optimezer
OPTIMIZER=SGD()
#hidden neurons
HIDDEN=128
#splitting the data in 20 percet tets data and rest for training
VALIDATION_SPLIT=0.2
img_rows, img_cols = 28, 28
#splitting the data into 60000 training data and 10000 testing data
(x_train, y_train),(x_test, y_test)=mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    Input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    Input_shape = (img_rows, img_cols, 1)


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
#normalising the value by dividing with 255

x_train /= 255
x_test /= 255

print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
#convert the class vector to binary class matrices

Y_train=np_utils.to_categorical(y_train,CLASSES)
Y_test=np_utils.to_categorical(y_test,CLASSES)


model=Sequential()
model.add(Conv2D(26,kernel_size=(3,3),activation='relu',input_shape=Input_shape))
model.add(Conv2D(26,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(HIDDEN,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(CLASSES,activation='softmax'))

json_file=open('mnist.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("mnist.h5")
print("loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

score=loaded_model.evaluate(x_test,Y_test,verbose=0)

print('accuracy: %.2f%%'%(score[1]*100))

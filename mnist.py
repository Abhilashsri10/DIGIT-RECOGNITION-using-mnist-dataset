# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:43:39 2018

@author: Abhilash Srivastava
"""

from __future__ import print_function
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

#for reproducing the random values
np.random.seed(1671)

#network setup
EPOCH=200
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
#splitting the data into 60000 training data and 10000 testing data
(x_train, y_train),(x_test, y_test)=mnist.load_data()

fig=plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i],cmap='gray',interpolation='none')
    plt.title('digit: {}'.format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig

#reshaped to 60000*784
RESHAPED=784
x_train=x_train.reshape(60000,RESHAPED)
x_test=x_test.reshape(10000,RESHAPED)
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
#setting up the network
model=Sequential()
model.add(Dense(HIDDEN,input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(HIDDEN))
model.add(Activation('relu'))
model.add(Dense(CLASSES))
model.add(Activation('softmax'))
model.summary()
#compiling the network
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
#training the model
data_train=model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
score=model.evaluate(x_test,y_test,verbose=VERBOSE)
print('TEST SCORE:',score[0])
print('test accuracy:',score[1])
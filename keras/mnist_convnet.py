#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/22 11:41
@File    : mnist_convnet.py
@Software: PyCharm
"""
from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical


def model_no_max_pool():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.Conv2D(64,(32,32),activation='relu'))


def conv_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()



train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model=conv_model()
print(model.summary())

#训练
model.fit(train_images,train_labels,epochs=5,batch_size=64)

#评估
test_loss,test_acc = model.evaluate(test_images,test_labels)
print('test_acc:',test_acc)


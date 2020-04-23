#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/23 9:54
@File    : vgg16.py
@Software: PyCharm
"""

from keras import models
from keras import layers

def vgg16(weights_path=None):
    model = models.Sequential()

    model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(256,256,3)))
    model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(4096,activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(4096,activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(1000,activation='softmax'))

    if  weights_path:
        model.load_weights(weights_path)

    return model

weight_path = r"C:\Users\123\.keras\datasets\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

model = vgg16(weight_path)
print(model.summary())

weight_con2d_2,bias_con2d_2 = model.get_layer('conv2d_2').get_weights()

print(weight_con2d_2.shape)
print(bias_con2d_2.shape)
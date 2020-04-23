#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/22 23:32
@File    : dog_cat_data_augment.py
@Software: PyCharm
"""

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

def dog_cat_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

    return model

train_dir =r"D:\文件\book\dogs-vs-cats\test_model\train"
validation_dir=r"D:\文件\book\dogs-vs-cats\test_model\validation"

#数据预处理
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
                                   height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,target_size=(150,150),batch_size=20,class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,target_size=(150,150),batch_size=20,class_mode='binary'
)

model =dog_cat_model()
# print(model.summary())

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,
                              validation_steps=50)

model.save('cat_dog_small_2.h5')
#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/6/5 14:01
@File    : GoogLeNet.py
@Software: PyCharm
"""
import keras
from keras import layers,Input,Model

class Inception_V1():
    pass

class Inception_V2():
    pass

class Inception_V3():
    pass

class Inception_V4():
    pass

def inception_v1(input,size):
    conv11,conv31,conv32,conv51,conv52,proj11=size

    ly_11 = layers.Conv2D(conv11, (1, 1), strides=1, activation='relu')(input)

    ly_3 = layers.Conv2D(conv31, (1, 1), strides=1, activation='relu')(input)
    ly_3 = layers.Conv2D(conv32, (3, 3), strides=1, activation='relu')(ly_3)

    ly_5 = layers.Conv2D(conv51, (1, 1), strides=1, activation='relu')(input)
    ly_5 = layers.Conv2D(conv52, (5, 5), strides=1, padding='same', activation='relu')(ly_5)

    proj = layers.MaxPooling2D((3, 3), strides=1, padding='same', activation='relu')(input)
    proj = layers.Conv2D(proj11, (1, 1), strides=1, activation='relu')(proj)

    out = layers.concatenate([ly_11, ly_3, ly_5, proj],axis=-1)


input = Input(shape=(224,224,3))

# #第一层
ly1 = layers.Conv2D(64,(7,7),strides=2,padding='same',activation='relu',)(input)
ly1 = layers.MaxPooling2D(64,(3,3),strides=2,padding='same',activatation='relu')(ly1)
#
# #第二层
ly2 = layers.Conv2D(64,(1,1),activation='relu')(ly1)
ly2 = layers.Conv2D(192,(3,3),strides=1,padding='same',activation='relu')(ly2)
ly2 = layers.MaxPooling2D(192,(3,3),strides=2,padding='same',activation='relu')(ly2)

# #第三层
# #3a层
ly3a_size = [64,96,128,16,32,32]
ly3a = inception_v1(ly2,ly3a_size)
#3b层
ly3b_size = [128,128,192,32,96,64]
ly3b=inception_v1(ly3a,ly3b_size)

ly3_pool = layers.MaxPooling2D((3,3),strides=2,padding='same',activation='relu')(ly3b)

#第4层
#4a
ly4a_size = [192,96,208,16,48,64]
ly4a = inception_v1(ly3_pool,ly4a_size)
#4b
ly4b_size =[160,112,224,24,64,64]
ly4b = inception_v1(ly4a,ly4b_size)
#4c
ly4c_size =[128,128,256,24,64,64]
ly4c = inception_v1(ly4b,ly4c_size)
#4d
ly4d_size =[112,144,288,32,64,64]
ly4d=inception_v1(ly4c,ly4d_size)
#4e
ly4e_size =[256,160,320,32,128,128]
ly4e =inception_v1(ly4d,ly4e_size)

ly4_pool = layers.MaxPooling2D((3,3),strides=2,padding='same',activation='relu')(ly4e)

#第5层
#5a
ly5a_size =[256,160,320,32,128,128]
ly5a=inception_v1(ly4_pool,ly5a_size)
#5b
ly5b_size =[384,192,384,48,128,128]
ly5b=inception_v1(ly5a,ly5b_size)

#第六层
ly6_pool = layers.AveragePooling2D((7,7),strides=1,padding='same',activation='relu')(ly5b)
flat = layers.Flatten()(ly6_pool)
ly6_pool=layers.Dropout(0.4)(ly6_pool)

ly6_fc = layers.Dense(1000,activation='softmax')(ly6_pool)

#局部分类网络
def exact_structure_part(input):

    o=layers.AveragePooling2D((5,5),strides=3,padding='valid',activation='relu')(input)
    o=layers.Conv2D(128,(1,1),strides=1,activation='liner')(o)
    o=layers.Dense(1024,activation='liner')(o)
    o=layers.Dropout(0.7)(o)
    o=layers.Dense(1000,activation='softmax')(o)

    return o

exact_structure_out1 = exact_structure_part(ly4a)
exact_structure_out2 = exact_structure_part(ly4d)

inception_model = Model(input,[ly6_fc,exact_structure_out1,exact_structure_out2])

inception_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'],
                        loss_weights=[0.6,0.2,0.2])















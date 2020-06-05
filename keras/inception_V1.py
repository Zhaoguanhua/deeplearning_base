#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/6/5 13:43
@File    : inception_V1.py
@Software: PyCharm
"""

from keras import layers,Input,Model

x=Input(shape=(256,256,3))

branch_a=layers.Conv2D(128,1,activation='relu',strides=2)(x)

branch_b=layers.Conv2D(128,1,activation='relu')(x)
branch_b = layers.Conv2D(128,3,activation='relu',strides=2)(branch_b)

branch_c =layers.AveragePooling2D(3,strides=2)(x)
branch_c = layers.Conv2D(128,3,activation='relu')(branch_c)

branch_d = layers.Conv2D(128,1,activation='relu')(x)
branch_d = layers.Conv2D(128,3,activation='relu')(branch_d)
branch_d = layers.Conv2D(128,3,activation='relu',strides=2)(branch_d)

out = layers.concatenate([branch_a,branch_b,branch_c,branch_d],aixs=-1)

model = Model(x,out)



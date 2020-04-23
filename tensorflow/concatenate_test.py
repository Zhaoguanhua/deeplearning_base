#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/23 13:23
@File    : concatenate_test.py
@Software: PyCharm
"""

import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf

t1=K.variable(np.array([[[1,2,3],[2,3,4]],[[4,4,5],[5,3,5]]]))
t2=K.variable(np.array([[[7,4,5],[8,4,1]],[[2,10,11],[15,11,-4]]]))
d0=K.concatenate([t1,t2],axis=0)
d1=K.concatenate([t1,t2],axis=1)
d2=K.concatenate([t1,t2],axis=2)
d3=K.concatenate([t1,t2],axis=2)
print(t1.shape)
print(d0.shape)
print(d1.shape)
print(d2.shape)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("t1")
    print(sess.run(t1))
    print("t2")
    print(sess.run(t2))
    print("d0")
    print(sess.run(d0))
    print("d1")
    print(sess.run(d1))
    print("d2")
    print(sess.run(d2))
    print("d3")
    print(sess.run(d3))
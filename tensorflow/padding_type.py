#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/22 16:00
@File    : padding_type.py
@Software: PyCharm
"""

import tensorflow as tf

x=tf.constant([[1.,2.,3.],[4.,5.,6.]])
print(x)
x=tf.reshape(x,[1,2,3,1])
print(x)

valid_pad = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
same_pad = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
#
# print(valid_pad.get_shape())
# print(same_pad.get_shape())

with tf.Session() as sess:
    print("image:")
    image=sess.run(x)
    print(image)
    print("result_valid:")
    result = sess.run(valid_pad)
    print(result)
    print("result_same:")
    result_same = sess.run(same_pad)
    print(result_same)

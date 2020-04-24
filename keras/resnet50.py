#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/23 14:06
@File    : resnet50.py
@Software: PyCharm
"""

import keras
#
#
# keras.utils.plot_model(keras.applications.ResNet50(include_top=True,input_shape=(224,224,3),
#                                                    weights=None))

from keras_applications import resnet,resnet_v2

resnet_v2.ResNet152V2(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 backend=keras.backend,
                 layers=keras.layers,
                 models=keras.models,
                 utils=keras.utils
                 )
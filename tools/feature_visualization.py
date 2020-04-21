#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/21 17:22
@File    : feature_visualization.py
@Software: PyCharm
"""

import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import numpy as np
import tensorflow as tf

model = VGG16(weights='imagenet', include_top=False)

img_path = 'data\elephants.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

features = np.mean(features, -1, keepdims=False)[0]

import cv2

features = cv2.resize(features, (224, 224), interpolation=cv2.INTER_LINEAR)

features /= np.max(features)

norm_img = np.zeros(features.shape)
# cv2.normalize(features, norm_img, 0, 255, cv2.NORM_MINMAX)
norm_img = np.asarray(features * 255, dtype=np.uint8)

heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
# heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像

origin = cv2.resize(cv2.imread(img_path), (224, 224))

img_add = cv2.addWeighted(origin, 0.3, heat_img, 0.7, 0)

img = cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB)

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

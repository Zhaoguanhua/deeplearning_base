#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/12 21:53
@File    : Visualization.py
@Software: PyCharm
"""

import numpy as np
from PIL import Image
import pickle
import os

CHANNEL = 3
WIDTH = 32
HEIGHT = 32

data = []
labels = []
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(5):
    with open(r"C:\Users\123\.keras\datasets\cifar-10-batches-py\data_batch_" + str(i + 1), mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data += list(data_dict[b'data'])
        labels += list(data_dict[b'labels'])

img = np.reshape(data, [-1, CHANNEL, WIDTH, HEIGHT])

data_path = r"D:\test\cifar"
if not os.path.exists(data_path):
    os.makedirs(data_path)
for i in range(img.shape[0]):
    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    name = "img-" + str(i) + "-" + classification[labels[i]] + ".png"
    rgb.save(os.path.join(data_path,name), "PNG")




#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/15 22:43
@File    : mnist_test.py
@Software: PyCharm
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

digit=train_images[5]
print(train_labels[5])
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

# network = models.Sequential()
# network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
# network.add(layers.Dense(10,activation='softmax'))
#
# network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#
# train_images = train_images.reshape((60000,28*28))
# train_images = train_images.astype('float32')/255
#
# test_images = test_images.reshape((10000,28*28))
# test_images = test_images.astype('float32')/255
#
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# #训练
# network.fit(train_images,train_labels,epochs=5,batch_size=128)
#
# #评估
# test_loss,test_acc = network.evaluate(test_images,test_labels)
# print('test_acc:',test_acc)






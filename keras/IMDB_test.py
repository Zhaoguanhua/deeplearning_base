#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/16 14:18
@File    : IMDB_test.py
@Software: PyCharm
"""
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.

    return results


(train_data,train_labels),(test_data,test_labels) = imdb.load_data(path=r"D:\迅雷下载\imdb.npz",num_words=10000)


# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# #标签向量化
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')
#
# #训练集和验证集
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
#
# #定义模型
# model = models.Sequential()
# model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
# model.add(layers.Dense(16,activation='relu'))
# model.add(layers.Dense(1,activation='sigmoid'))
#
# #编译模型
# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#
# #训练模型
# history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
#
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1,len(loss_values)+1)
#
# #绘制训练损失和验证损失
# plt.plot(epochs,loss_values,'r',label='Training loss')
# plt.plot(epochs,val_loss_values,'b',label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# #绘制训练精度和验证精度
# plt.clf()
# acc=history_dict['acc']
# val_acc=history_dict['val_acc']
#
# plt.plot(epochs,acc,'g',label='Training acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


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
from keras import regularizers
import matplotlib.pyplot as plt

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.

    return results

def base_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def Smaller_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return  model

def L2_model():
    model = models.Sequential()
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
                           activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),
                           activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def L1_model():
    model = models.Sequential()
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l1(0.001),
                           activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l1(0.001),
                           activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def L1_L2_model():
    model = models.Sequential()
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l1_l2(0.001),
                           activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16,kernel_regularizer=regularizers.l1_l2(0.001),
                           activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def Dropout_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#训练集和验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#定义模型1
model = base_model()
#编译模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#训练模型
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

#***********************************************************************************************************
#测试减少网络规模
#定义模型2
# model2=Smaller_model()
# #训练模型2
# history2 = model2.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# history_dict2 = history2.history
# loss_values2 = history_dict2['loss']
# val_loss_values2 = history_dict2['val_loss']

#***********************************************************************************************************
#测试L2权重正则化
#定义模型3
# model3 = L2_model()
# #训练模型3
# history3 = model3.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# history_dict3 = history3.history
# loss_values3 = history_dict3['loss']
# val_loss_values3 = history_dict3['val_loss']

#***********************************************************************************************************
#测试L1权重正则化
#定义模型4
# model4=L1_model()
# #训练模型4
# history4 = model4.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# history_dict4 = history4.history
# loss_values4 = history_dict4['loss']
# val_loss_values4 = history_dict4['val_loss']

#***********************************************************************************************************
#测试L1_L2权重正则化
#定义模型5
# model5=L1_L2_model()
# #训练模型4
# history5 = model5.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# history_dict5 = history5.history
# loss_values5 = history_dict5['loss']
# val_loss_values5 = history_dict5['val_loss']

#***********************************************************************************************************
#测试L1_L2权重正则化
#定义模型6
model6=Dropout_model()
#训练模型4
history6 = model6.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
history_dict6 = history6.history
loss_values6 = history_dict6['loss']
val_loss_values6 = history_dict6['val_loss']

epochs = range(1,len(loss_values)+1)
#对比两种模型的验证损失
plt.plot(epochs,val_loss_values,'b',label='Original model')
plt.plot(epochs,val_loss_values6,'r',label='Dropout-regularized model')
plt.title('validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# #绘制训练损失和验证损失
# plt.plot(epochs,loss_values,'r',label='Training loss')
# plt.plot(epochs,val_loss_values,'b',label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

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


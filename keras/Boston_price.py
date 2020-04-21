#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : zhaogh@hdsxtech.com
@Time    : 2020/4/13 18:11
@File    : Boston_price.py
@Software: PyCharm
"""

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.datasets import boston_housing
import numpy as np

def build_model():
    model=Sequential()
    model.add(Dense(64,activation='relu',input_dim=13))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    #print(x_train.shape)
    mean = x_train.mean(axis=0)
    x_train-=mean
    std=x_train.std(axis=0)
    x_train /=std
    x_test-=mean
    x_test/=std

    k = 4
    num_val_samples = len(x_train) // k
    num_epochs = 100
    all_scores = []
    for i in range(k):
      print('processing fold #', i)
      # Prepare the validation data: data from partition # k
      val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
      #print(val_data.shape)
      val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
      # # Prepare the training data: data from all other partitions
      # print(x_train[:i * num_val_samples].shape)
      # print(x_train[(i + 1) * num_val_samples:].shape)
      partial_train_data = np.concatenate([x_train[:i * num_val_samples],x_train[(i + 1) * num_val_samples:]],axis=0)
      partial_train_targets = np.concatenate([y_train[:i * num_val_samples],y_train[(i + 1) * num_val_samples:]],axis=0)
      # Build the Keras model (already compiled)
      model = build_model()
      # Train the model (in silent mode, verbose=0)
      model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
      # Evaluate the model on the validation data
      val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
      all_scores.append(val_mae)

    print(all_scores)
    print(np.mean(all_scores))
# print(x_train[0],y_train[0])
# print(x_test[0],y_test[0])
#
# model = Sequential()
# #设置全连接层，输入为13 输入为1
# model.add(Dense(5,input_dim=13))
# print(model.layers[0].get_weights())
#
# model.add(Activation('relu'))
# model.add(Dense(1))
#
# #编译模型
# model.compile(loss='mean_squared_error',optimizer='sgd')
# #训练模型
# print('Training -----------')
# for step in range(301):
#     cost = model.train_on_batch(x_train, y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
#     if step % 100 == 0:
#         print('train cost: ', cost)



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
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if  smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

def build_model():
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(x_train.shape[1],)))
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
    num_epochs = 500
    #all_scores = []
    all_mae_histories =[]
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
      history=model.fit(partial_train_data, partial_train_targets,validation_data=(val_data,val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
      print(history.history.keys())
      mae_history = history.history['val_mean_absolute_error']
      all_mae_histories.append(mae_history)
      # # Evaluate the model on the validation data
      # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
      # all_scores.append(val_mae)

    # print(all_scores)
    # print(np.mean(all_scores))

    average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel('Epoches')
    plt.ylabel('Validation MAE')
    plt.show()

    smooth_mae_history = smooth_curve(average_mae_history[10:])

    plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()



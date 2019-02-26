# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:13:03 2019

CNN for predict the B vector from the simulated images

simple network structure version

contains two convolutional layers with max pooling

followed by two fully connected layers

ATTENTION: commend out the save model and file lines before running the script

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import load_model

image_res = 64
num_total_sample = 8000

regressor = Sequential()

regressor.add(Convolution2D(64, (3, 3), input_shape = (image_res, image_res, 1), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Convolution2D(32, (3, 3), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Flatten())

regressor.add(Dense(512, activation = 'relu'))

regressor.add(Dense(44))

regressor.compile('adam', loss = 'mean_squared_error')

X_data = np.load(r'D:\irimages\irholography\CNN\data_v7_padded\im_data_intensity.npy')
X_data = np.reshape(X_data, (image_res, image_res, 1, num_total_sample))
X_data = np.swapaxes(X_data, 0, -1)
X_data = np.swapaxes(X_data, -2, -1)

y_data_real = np.load(r'D:\irimages\irholography\CNN\data_v7_padded\B_data_real.npy')
y_data_imag = np.load(r'D:\irimages\irholography\CNN\data_v7_padded\B_data_imag.npy')

y_data = np.concatenate((y_data_real, y_data_imag), axis = 0)

y_data = np.reshape(y_data, (44, num_total_sample))
y_data = np.swapaxes(y_data, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)

regressor.fit(x = X_train, y = y_train, batch_size = 50,
              epochs = 25,
              validation_split = 0.2)

y_pred = regressor.predict(X_test)



#
###
##de-normalize features
#for i in range(3):
#    y_pred[:, i] = y_pred[:, i] * y_var[i] + y_mean[i]

#y_pred[:, 0] /= 10
#y_test[:, 0] /= 10
#
#y_pred[:, 2] *= 10
#y_test[:, 2] *= 10

#plt.figure()
#plt.plot(y_test[0, :])
#plt.plot(y_pred[0, :])
#
#plt.plot(y_test[1, :])
#plt.plot(y_pred[2, :])

#
#plt.figure()
#plt.subplot(3,1,1)
#plt.plot(y_test[::10,0], label = 'Ground Truth')
#plt.plot(y_pred[::10,0], linestyle='dashed', label = 'Prediction')
#plt.legend()
#plt.title('Real Part')
#
#plt.subplot(3,1,2)
#plt.plot(y_test[::10,1], label = 'Ground Truth')
#plt.plot(y_pred[::10,1], linestyle='dashed', label = 'Prediction')
#plt.legend()
#plt.title('Imaginary Part')
#
#plt.subplot(3,1,3)
#plt.plot(y_test[::10,2], label = 'Ground Truth')
#plt.plot(y_pred[::10,2], linestyle='dashed', label = 'Prediction')
#plt.legend()
#plt.title('Radius')
#
#from tensorflow.python.client import device_lib
#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    
#    return[x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
#get_available_gpus()
#
regressor.save(r'D:\irimages\irholography\CNN\CNN_v9_bandpass\intensity_simple.h5')
###
np.save(r'D:\irimages\irholography\CNN\CNN_v9_bandpass\X_test.bin', X_test)
np.save(r'D:\irimages\irholography\CNN\CNN_v9_bandpass\y_test.bin', y_test)
#
#
##
#
#cnnv7 = load_model(r'D:\irimages\irholography\CNN\CNN_v5_B_from_ES\model_v1.h5')
#X_test = np.load(r'D:\irimages\irholography\CNN\CNN_v5_B_from_ES\X_test.bin.npy')
#y_test = np.load(r'D:\irimages\irholography\CNN\CNN_v5_B_from_ES\y_test.bin.npy')
#
##
#y_pred = cnnv7.predict(X_test)
#
#y_pred_w_noise = cnnv7.predict(X_test_w_noise)
#
#x_index = np.arange(0, np.size(y_test[::20,0]), 1)
#
#plt.figure()
#plt.subplot(3,1,1)
#plt.scatter(x_index, y_test[::20,0], s = 2, label = 'Ground Truth')
#plt.scatter(x_index, y_pred[::20,0], s = 2, label = 'Prediction')
#plt.legend()
#plt.title('Real Part')
#
#plt.subplot(3,1,2)
#plt.scatter(x_index, y_test[::20,1], s = 2, label = 'Ground Truth')
#plt.scatter(x_index, y_pred[::20,1], s = 2, label = 'Prediction')
#plt.legend()
#plt.title('Imaginary Part')
#
#plt.subplot(3,1,3)
#plt.scatter(x_index, y_test[::20,2], s = 2, label = 'Ground Truth')
#plt.scatter(x_index, y_pred[::20,2], s = 2, label = 'Prediction')
#plt.legend()
#plt.title('Radius')
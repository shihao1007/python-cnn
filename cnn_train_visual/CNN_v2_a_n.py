# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:06:14 2019

second edition of the CNN for a and n
data is preprocessed

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

image_res = 128
num = 20
num_samples = num ** 3

#%%
# define CNN structure

regressor = Sequential()

regressor.add(Convolution2D(128, (3, 3), input_shape = (image_res, image_res, 1), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Convolution2D(64, (3, 3), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Convolution2D(64, (3, 3), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))
#
#regressor.add(Convolution2D(64, (3, 3), activation = 'relu'))
#
#regressor.add(MaxPooling2D(pool_size = (2, 2)))
#
##regressor.add(MaxPooling2D(pool_size = (2, 2)))
#
#regressor.add(Convolution2D(32, (3, 3), activation = 'relu'))
#
#regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Flatten())

regressor.add(Dense(128, activation = 'relu'))

regressor.add(Dense(3))

regressor.compile('adam', loss = 'mean_squared_error')

#%%
# load data set

X_train = np.load(r'D:\irimages\irholography\CNN\data_v10_far_field\split_data\train\cropped_X_train_intensity.npy')

X_test = np.load(r'D:\irimages\irholography\CNN\data_v10_far_field\split_data\test\cropped_X_test_intensity.npy')

y_train = np.load(r'D:\irimages\irholography\CNN\data_v10_far_field\split_data\train\y_train.npy')

y_test = np.load(r'D:\irimages\irholography\CNN\data_v10_far_field\split_data\test\y_test.npy')

#%%
# scale the attenuation coefficient feature

y_train[:, 1] *= 100
#%%
# train the network

regressor.fit(x = X_train, y = y_train, batch_size = 50,
              epochs = 25,
              validation_split = 0.2)

#%%
# get the prediction from the network
y_pred = regressor.predict(X_test)

# down scale it
y_pred[:, 1] /= 100

y_off = np.abs(y_pred - y_test)
y_off_perc = np.mean(y_off/[2.0, 0.05, 2.0], axis = 0) * 100

print('Current Model:')
print('Refractive Index (Real) Error: ' + str(y_off_perc[0]) + ' %')
print('Refractive Index (Imaginary) Error: ' + str(y_off_perc[1]) + ' %')
print('Redius of the Sphere Error: ' + str(y_off_perc[2]) + ' %')

#%%
# plot the prediction and ground truth
plt.figure()
plt.subplot(3,1,1)
plt.plot(y_test[::32,0], label = 'Ground Truth')
plt.plot(y_pred[::32,0], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Real Part')

plt.subplot(3,1,2)
plt.plot(y_test[::32,1], label = 'Ground Truth')
plt.plot(y_pred[::32,1], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Imaginary Part')

plt.subplot(3,1,3)
plt.plot(y_test[::32,2], label = 'Ground Truth')
plt.plot(y_pred[::32,2], linestyle='dashed', label = 'Prediction')
plt.legend()
plt.title('Radius')
#
#from tensorflow.python.client import device_lib
#def get_available_gpus():
#    local_device_protos = device_lib.list_local_devices()
#    
#    return[x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
#get_available_gpus()

regressor.save(r'D:\irimages\irholography\CNN\CNN_v12_far_field_a_n\intensity\intensity_v0.h5')

#from keras.models import load_model
#cnnv7 = load_model(r'D:\irimages\irholography\CNN\CNN_v2\model_v7_best.h5')
#
#y_pred = cnnv7.predict(X_test)
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
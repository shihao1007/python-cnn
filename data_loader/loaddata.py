# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:53:15 2019

@author: shihao

Load data set for training

"""

import numpy as np

image_res = 128
num_total_sample = 15625

X_data = np.load(r'D:\irimages\irholography\CNN\data_v2\im_data.bin.npy')
X_data = np.reshape(X_data, (image_res, image_res, 2, num_total_sample))
X_data = np.swapaxes(X_data, 0, -1)
X_data = np.swapaxes(X_data, -2, -1)

y_data = np.load(r'D:\irimages\irholography\CNN\data_v2\lable_data.bin.npy')
y_data = np.reshape(y_data, (3, num_total_sample))
y_data = np.swapaxes(y_data, 0, 1)
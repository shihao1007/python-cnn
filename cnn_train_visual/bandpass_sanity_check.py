# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:15:54 2019

bandpass sanity check

Editor:
    Shihao Ran
    STIM Laboratory
"""

X_test = np.load(r'D:\irimages\irholography\CNN\data_v9_far_field\split_data\test\im_data_intensity_0.9.npy')
y_pred = intensity_CNN.predict(X_test)

y_pred[:, 1] /= 100

y_off = y_test - y_pred

y_off_perc = np.abs(np.average(y_off / y_test, axis = 0) * 100)

print('Refractive Index (Real) Error: ' + str(y_off_perc[0]) + ' %')
print('Refractive Index (Imaginary) Error: ' + str(y_off_perc[1]) + ' %')
print('Redius of the Sphere Error: ' + str(y_off_perc[2]) + ' %')
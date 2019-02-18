# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:29:20 2019

@author: shihao

prediction evaluation
"""

#
##evaluation for a and n
#import numpy as np
#
#y_off = y_test - y_pred
#
#y_off_perc = np.abs(np.average(y_off / y_test, axis = 0) * 100)
#
#print('Refractive Index (Real) Error: ' + str(y_off_perc[0]) + ' %')
#print('Refractive Index (Imaginary) Error: ' + str(y_off_perc[1]) + ' %')
#print('Redius of the Sphere Error: ' + str(y_off_perc[2]) + ' %')


#evaluation for B

#y_off = y_test - y_pred_w_noise
y_off = y_test - y_pred
y_off_sum = np.sum(y_off, axis = 1)

y_test_sum = np.sum(y_test, axis = 1)

y_off_ratio = y_off_sum / y_test_sum

y_off_perc = np.abs(np.average(y_off_ratio) * 100)

print('Relative B Error (Vector Sum): ' + str(y_off_perc) + ' %')


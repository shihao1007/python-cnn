# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:40:27 2019

load sequential data into memory and crop the image so it can be saved in a training data set

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
from matplotlib import pyplot as plt
import sys
    
def cropImage(Etot, res):
    
    #initialize cropping
    cropsize = res
    startIdx = int(np.fix(simRes /2 + 1) - np.floor(cropsize/2))
    endIdx = int(startIdx + cropsize - 1)
    
    D_Et = np.zeros((cropsize, cropsize), dtype = np.complex128)
    D_Et = Etot[startIdx:endIdx+1, startIdx:endIdx+1]
#    D_Ef = np.zeros((cropsize, cropsize), dtype = np.complex128)
#    D_Ef = Ef_bpf[startIdx:endIdx, startIdx:endIdx]

    return D_Et

# specify parameters of the simulation
res = 128
padding = 2
fov = 30
lambDa = 2 * np.pi
halfgrid = np.ceil(fov / 2) * (padding * 2 + 1)

# get the resolution after padding the image
simRes = res * (padding *2 + 1)

# dimention of the data set
nb_a = 20
nb_nr = 20
nb_ni = 20

nb_img = nb_a * nb_nr * nb_ni

# parent directory of the data set
data_dir = r'D:\irimages\irholography\CNN\data_v8_padded'

# allocate space for the image data set
im_cropped_complex = np.zeros((res, res, 2, nb_nr, nb_ni, nb_a))
im_cropped_intensity = np.zeros((res, res, 1, nb_nr, nb_ni, nb_a))

cnt = 0
# band pass and crop
for h in range(nb_a):
    sphere_dir = data_dir + '\im_data%3.3d'% (h) + '.npy'
    sphere_data = np.load(sphere_dir)
    
    complex_im = sphere_data[:,:,0,:,:] + sphere_data[:,:,1,:,:] * 1j
    
    for i in range(nb_nr):
        for j in range(nb_ni):
            cropped_im_complex = cropImage(complex_im[:, :, i, j], res)
            im_cropped_complex[:, :, 0, i, j, h] = np.real(cropped_im_complex)
            im_cropped_complex[:, :, 1, i, j, h] = np.imag(cropped_im_complex)
            
            im_cropped_intensity[:, :, 0, i, j, h] = np.abs(cropped_im_complex) ** 2
            
            # print progress
            cnt += 1
            sys.stdout.write('\r' + str(cnt / nb_img * 100)  + ' %')
            sys.stdout.flush() # important

np.save(data_dir + '\cropped_data_complex', im_cropped_complex)
np.save(data_dir + '\cropped_data_intensity', im_cropped_intensity)



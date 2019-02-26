# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:40:27 2019

load sequential data into memory and apply a bandpass filter to the images

then crop the image smaller so the total training set won't reach memory limit

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
from matplotlib import pyplot as plt
import sys

def BPF(halfgrid, simRes, NA_in, NA_out):
    #create a bandpass filter
        #change coordinates into frequency domain
        
    df = 1/(halfgrid*2)
    
    iv, iu = np.meshgrid(np.arange(0, simRes, 1), np.arange(0, simRes, 1))
    
    u = np.zeros(iu.shape)
    v = np.zeros(iv.shape)
    
    #initialize the filter as All Pass
    BPF = np.ones(iv.shape)
    
    idex1, idex2 = np.where(iu <= simRes/2)
    u[idex1, idex2] = iu[idex1, idex2]
    
    idex1, idex2 = np.where(iu > simRes/2)
    u[idex1, idex2] = iu[idex1, idex2] - simRes +1
    
    u *= df
    
    idex1, idex2 = np.where(iv <= simRes/2)
    v[idex1, idex2] = iv[idex1, idex2]
    
    idex1, idex2 = np.where(iv > simRes/2)
    v[idex1, idex2] = iv[idex1, idex2] - simRes +1
    
    v *= df
    
    magf = np.sqrt(u ** 2 + v ** 2)
    
    #block lower frequency
    idex1, idex2 = np.where(magf < NA_in / lambDa)
    BPF[idex1, idex2] = 0
    #block higher frequency
    idex1, idex2 = np.where(magf > NA_out / lambDa)
    BPF[idex1, idex2] = 0
    
    return BPF
    
def imgAtDetec(Etot, bpf):
    #2D fft to the total field
    Et_d = np.fft.fft2(Etot)
#    Ef_d = np.fft.fft2(Ef)
    
    #apply bandpass filter to the fourier domain
    Et_d *= bpf
#    Ef_d *= bpf
    
    #invert FFT back to spatial domain
    Et_bpf = np.fft.ifft2(Et_d)
#    Ef_bpf = np.fft.ifft2(Ef_d)
    
    #initialize cropping
    cropsize = res
    startIdx = int(np.fix(simRes /2 + 1) - np.floor(cropsize/2))
    endIdx = int(startIdx + cropsize - 1)
    
    D_Et = np.zeros((cropsize, cropsize), dtype = np.complex128)
    D_Et = Et_bpf[startIdx:endIdx+1, startIdx:endIdx+1]
#    D_Ef = np.zeros((cropsize, cropsize), dtype = np.complex128)
#    D_Ef = Ef_bpf[startIdx:endIdx, startIdx:endIdx]

    return D_Et

# specify parameters of the simulation
res = 128
padding = 2
fov = 30
lambDa = 2 * np.pi
halfgrid = np.ceil(fov / 2) * (padding * 2 + 1)
NA_in = 0.0
NA_out = 0.9

# get the resolution after padding the image
simRes = res * (padding *2 + 1)

# calculate the band pass filter
bpf = BPF(halfgrid, simRes, NA_in, NA_out)

# dimention of the data set
nb_a = 20
nb_nr = 20
nb_ni = 20

nb_img = nb_a * nb_nr * nb_ni

# parent directory of the data set
data_dir = r'D:\irimages\irholography\CNN\data_v8_padded'

# allocate space for the image data set
im_data_complex = np.zeros((res, res, 2, nb_nr, nb_ni, nb_a))
im_data_intensity = np.zeros((res, res, 1, nb_nr, nb_ni, nb_a))

cnt = 0
# band pass and crop
for h in range(nb_a):
    sphere_dir = data_dir + '\im_data%3.3d'% (h) + '.npy'
    sphere_data = np.load(sphere_dir)
    
    complex_im = sphere_data[:,:,0,:,:] + sphere_data[:,:,1,:,:] * 1j
    intensity_im = np.abs(complex_im) ** 2
    
    for i in range(nb_nr):
        for j in range(nb_ni):
            filtered_im_complex = imgAtDetec(complex_im[:, :, i, j], bpf)
            im_data_complex[:, :, 0, i, j, h] = np.real(filtered_im_complex)
            im_data_complex[:, :, 1, i, j, h] = np.imag(filtered_im_complex)
            
            filtered_im_intensity = imgAtDetec(intensity_im[:, :, i, j], bpf)
            im_data_intensity[:, :, 0, i, j, h] = np.abs(filtered_im_intensity) ** 2
            
            # print progress
            cnt += 1
            sys.stdout.write('\r' + str(cnt / nb_img * 100)  + ' %')
            sys.stdout.flush() # important

np.save(data_dir + '\im_data_complex_0.9', im_data_complex)
np.save(data_dir + '\im_data_intensity_0.9', im_data_intensity)


